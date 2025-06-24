"""
OpenAI Agents SDK execution engine implementation.
Refactored internally to a functional style with a typed @dataclass context,
while preserving the original public API.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agent_workflow.providers import FunctionTool, OpenAITool, ToolRegistry

from agent_workflow.providers.mcp import MCPServerRegistry
from agent_workflow.providers.llm_tracing_utils import LLMTracer
from agent_workflow.providers.llm_observability import TraceStatus
# Using LLMTracer now
from agent_workflow.workflow_engine.execution_engines.base import (
    ExecutionEngine,
    ExecutionEngineFactory,
    ProgressCallback,
)
from agent_workflow.workflow_engine.execution_engines.openai.OpenAIAgentAdapter import (
    OpenAIAgentAdapter,
)
from agent_workflow.workflow_engine.execution_engines.openai.openai_hooks import OpenAIHooks
from agent_workflow.workflow_engine.models import (
    AgentOutput,
    BaseProviderConfig,
    ExecutionResult,
    LLMAgent,
    MCPServerType,
    ResponseStore,
    StageExecutionResult,
    StageExecutionStrategy,
    TaskExecutionResult,
    Workflow,
    WorkflowInput,
    WorkflowStage,
    WorkflowTask,
    ModelSettings,
)


logger = logging.getLogger("workflow-engine.execution_engine.openai")


# ----------------------------------------------------------------------------
# 1) Define a typed @dataclass context instead of a raw dict
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineContext:
    Agent: Any
    Runner: Any
    function_tool: Callable
    MCPServerSse: Any
    MCPServerStdio: Any
    AgentsModelSettings: Any
    tool_registry: Optional[ToolRegistry]
    mcp_registry: Optional[MCPServerRegistry]
    response_store: ResponseStore
    llm_tracer: LLMTracer
    hooks: OpenAIHooks



# ----------------------------------------------------------------------------
# 2) Build and return EngineContext (replaces __init__)
# ----------------------------------------------------------------------------
def init_context(
    tool_registry: ToolRegistry,
    mcp_registry: MCPServerRegistry,
    llm_tracer: LLMTracer,

) -> EngineContext:
    try:
        from agents import Agent, Runner, function_tool, ModelSettings as AgentsModelSettings
        from agents.mcp import MCPServerSse, MCPServerStdio
        logger.info("Using the OpenAI Agents SDK")
    except ImportError:
        logger.error(
            "OpenAI Agents SDK not installed. Install it with: pip install openai-agents"
        )
        raise

    if tool_registry is None:
        try:
            from agent_workflow.providers import global_tool_registry as global_registry
            tool_registry = global_registry
            logger.info("Using global tool registry")
        except ImportError:
            logger.warning("No tool registry provided nor found globally")
            raise

    if mcp_registry is None:
        try:
            from agent_workflow.providers.mcp import mcp_registry as global_mcp_registry
            mcp_registry = global_mcp_registry
            logger.info("Using global MCP server registry")
        except ImportError:
            logger.warning("No MCP server registry provided nor found globally")
            raise

    response_store = ResponseStore()

    return EngineContext(
        Agent=Agent,
        Runner=Runner,
        function_tool=function_tool,
        MCPServerSse=MCPServerSse,
        MCPServerStdio=MCPServerStdio,
        AgentsModelSettings=AgentsModelSettings,
        tool_registry=tool_registry,
        mcp_registry=mcp_registry,
        response_store=response_store,
        llm_tracer=llm_tracer,
        hooks=OpenAIHooks(llm_tracer)
    )


# ----------------------------------------------------------------------------
# 3) Pure helper functions
# ----------------------------------------------------------------------------

def _safe_serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)


def _provider_model(ctx: EngineContext, provider: BaseProviderConfig) -> Any:
    if provider is None:
        logger.error("Provider configuration is required")
        raise ValueError("Provider configuration is required")
    adapter = OpenAIAgentAdapter(
        provider=provider,
        base_url=provider.base_url,
        api_key=provider.api_key,
    )
    return adapter.chat_completion_model(provider.model)


def _create_model_settings(
    ctx: EngineContext, settings: ModelSettings
) -> Any:
    AMS = ctx.AgentsModelSettings
    if not settings:
        logger.warning("No model settings provided, using defaults")
        return AMS()
    return AMS(
        temperature=settings.temperature,
        top_p=settings.top_p,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        tool_choice=settings.tool_choice,
        parallel_tool_calls=settings.parallel_tool_calls,
        truncation=settings.truncation,
        max_tokens=settings.max_tokens,
        store=settings.store,
        include_usage=settings.include_usage,
    )


def _create_tool(ctx: EngineContext, tool_def: Dict[str, Any]) -> Any:
    name = tool_def.get("name")
    if not name:
        logger.warning("Tool definition missing name, skipping")
        return None
    registry = ctx.tool_registry
    tool = registry.get_tool(name) if registry else None
    if isinstance(tool, FunctionTool) and tool.function:
        # update the name of method as user would be referring it with the name
        import functools
        original_func = tool.function
        @functools.wraps(original_func)
        async def wrapped(*args, **kwargs):
            import inspect
            result = original_func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        wrapped.__name__ = name
        return ctx.function_tool(wrapped)
    if isinstance(tool, OpenAITool) and tool.function:
        return tool.function
    logger.warning(f"Tool {name} not usable by OpenAI Agents SDK, skipping")
    return None


async def _create_mcp_server(ctx: EngineContext, server_name: str) -> Any:
    registry = ctx.mcp_registry
    config = registry.get_server(server_name) if registry else None
    if not config:
        logger.warning(f"No MCP server named {server_name}, skipping")
        return None
    if config.server_type is MCPServerType.STDIO:
        server = ctx.MCPServerStdio(
            name=config.name,
            params=config.params,
            cache_tools_list=config.cache_tools_list,
            client_session_timeout_seconds=config.client_session_timeout,
        )
    else:
        server = ctx.MCPServerSse(
            name=config.name,
            params=config.params,
            cache_tools_list=config.cache_tools_list,
            client_session_timeout_seconds=config.client_session_timeout,
        )

    try:
        await server.connect()
    except Exception as e:
        logger.error(f"Error connecting to MCP server named {server_name}, skipping. Error - {e}")
        return None

    return server


def _create_prompt(
    ctx: EngineContext,
    task: WorkflowTask,
    inputs: WorkflowInput,
    processed_inputs: Dict[str, Any],
    append_output_structure: bool = False,
) -> str:
    def post(p: str) -> str:
        if append_output_structure:
            p += f"\n## Output Structure:\n{json.dumps(task.agent.output_schema)}"
        return p
    from jinja2 import BaseLoader, Environment
    env = Environment(loader=BaseLoader())
    try:
        tpl = env.from_string(task.prompt)
        data = processed_inputs or inputs.workflow or {}
        return post(tpl.render(**data))
    except Exception:
        data = processed_inputs or inputs.workflow or {}
        prompt = ""
        for k, v in data.items():
            if isinstance(v, dict):
                val = json.dumps(v, indent=2)
            elif isinstance(v, list):
                val = (
                    "\n".join(f"- {i}" for i in v)
                    if all(isinstance(i, str) for i in v)
                    else json.dumps(v, indent=2)
                )
            else:
                val = str(v)
            prompt += f"## {k}\n{val}\n\n"
        return post(prompt)


async def _create_agent_from_plan_agent(
    ctx: EngineContext, agent_cfg: LLMAgent, provider: BaseProviderConfig
) -> Any:
    name = agent_cfg.name or agent_cfg.id
    if provider is None:
        raise ValueError(f"Provider is required for agent {name}")
    instructions = agent_cfg.system_prompt or f"You are {name}, an assistant."
    tools = [_create_tool(ctx, t.dict()) for t in (agent_cfg.tools or [])]
    mcp_servers = [await _create_mcp_server(ctx, s) for s in (agent_cfg.mcp_servers or [])]
    output_type = agent_cfg.pydantic_output_schema()
    AgentClass = ctx.Agent
    model = _provider_model(ctx, provider)
    model_settings = _create_model_settings(ctx, provider.model_settings)
    if output_type and provider.enforce_structured_output:
        return AgentClass(
            name=name,
            instructions=instructions,
            model=model,
            tools=[t for t in tools if t],
            output_type=output_type,
            mcp_servers=[s for s in mcp_servers if s],
            model_settings=model_settings,
        )
    return AgentClass(
        name=name,
        instructions=instructions,
        model=model,
        tools=[t for t in tools if t],
        mcp_servers=[s for s in mcp_servers if s],
        model_settings=model_settings,
    )


async def _create_agent_from_task(
    ctx: EngineContext, task: WorkflowTask
) -> Any:
    if not task.agent:
        raise ValueError(f"Task {task.name} missing agent config")
    cfg = task.agent
    plan = LLMAgent(
        id=cfg.id,
        name=cfg.name or task.name,
        description=cfg.description or task.description,
        version=cfg.version,
        agent_type=cfg.agent_type,
        tools=cfg.tools,
        mcp_servers=cfg.mcp_servers,
        input_schema=cfg.input_schema,
        output_schema=cfg.output_schema,
        system_prompt=cfg.system_prompt,
        user_prompt=task.prompt,
        resources=cfg.resources,
        retry=cfg.retry,
    )
    return await _create_agent_from_plan_agent(ctx, plan, task.provider)

# ----------------------------------------------------------------------------
# 4) Core execution functions
# ----------------------------------------------------------------------------

async def initialize_workflow(
    ctx: EngineContext,
    workflow: Workflow,
    progress_callback: Optional[ProgressCallback] = None,
) -> Workflow:

    name = workflow.name

    if progress_callback:
        await progress_callback.on_workflow_start(name, workflow)

    for stage in workflow.stages:
        for task in stage.tasks:
            try:
                agent = await _create_agent_from_task(ctx, task)
                task.initialized_agent = agent
                logger.info(f"Initialized agent for task {task.name}")
            except Exception as e:
                logger.error(f"Init error for task {task.name}: {e}")

    if progress_callback:
        await progress_callback.on_workflow_complete(name, {"action": "initialize", "status": "complete"})
    return workflow


async def execute_workflow(
    ctx: EngineContext,
    workflow: Workflow,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> ExecutionResult:

    name = workflow.name
    if progress_callback:
        await progress_callback.on_workflow_start(name, workflow)
        
    # Start workflow trace group
    workflow_id = getattr(workflow, 'id', name)
    group_id = await ctx.llm_tracer.start_workflow_tracing(
        workflow_id=workflow_id,
        workflow_name=name
    )

    # Setup runner and pass the group_id to execute_stage
    runner = ctx.Runner()
    kwargs.setdefault("runner", runner)
    
    start_time = time.time()
    try:
        for stage in workflow.stages:
            if progress_callback:
                await progress_callback.on_stage_start(stage.name, stage)
            await execute_stage(ctx, stage, inputs, progress_callback, kwargs)
            if progress_callback:
                # no need to pass full dict twice
                await progress_callback.on_stage_complete(stage.name, {} )
        
        agent_outputs = []
        for stg, tasks in ctx.response_store.responses.items():
            for tname, tres in tasks.items():
                agent_outputs.append(AgentOutput(agent=tname, output=tres.result, metadata={"stage": stg}))
        
        all_agents = [t.name for s in workflow.stages for t in s.tasks]
        final_output = ""
        if agent_outputs:
            last = agent_outputs[-1].output
            if isinstance(last, dict):
                for key in ("result", "response", "output", "answer", "summary"):
                    if key in last:
                        final_output = last[key]
                        break
                if not final_output:
                    final_output = json.dumps(last)
            else:
                final_output = str(last)
        
        result = ExecutionResult(
            agent_outputs=agent_outputs,
            final_result=final_output,
            all_agents=all_agents,
            metadata={"workflow_name": name},
            response_store=ctx.response_store,
        )
        
        # End workflow trace group
        duration_ms = int((time.time() - start_time) * 1000)
        await ctx.llm_tracer.end_workflow_tracing(
            group_id=group_id,
            status=TraceStatus.SUCCESS,
            metadata={"tags": {"duration_ms": str(duration_ms)}}
        )
        
        if progress_callback:
            await progress_callback.on_workflow_complete(name, result)
            
        return result
        
    except Exception as e:
        # End workflow trace group with error
        duration_ms = int((time.time() - start_time) * 1000)
        await ctx.llm_tracer.end_workflow_tracing(
            group_id=group_id,
            status=TraceStatus.FAILED,
            error=e,
            metadata={"tags": {"duration_ms": str(duration_ms)}}
        )
        raise


async def execute_stage(
    ctx: EngineContext,
    stage: WorkflowStage,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback],
    kwargs: Dict[str, Any],
) -> StageExecutionResult:
    # Get workflow_group_id from kwargs
    try:
        et = stage.execution_type
        result = None
        
        if et is StageExecutionStrategy.SEQUENTIAL:
            result = await execute_stage_sequential(ctx, stage, inputs, progress_callback, kwargs)
        elif et is StageExecutionStrategy.PARALLEL:
            result = await execute_stage_parallel(ctx, stage, inputs, progress_callback, kwargs)
        elif et is StageExecutionStrategy.HANDOFF:
            result = await execute_stage_handoff(ctx, stage, inputs, progress_callback, kwargs)
        
        return result
        
    except Exception as e:
        raise

async def execute_stage_sequential(
    ctx: EngineContext,
    stage: WorkflowStage,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback],
    kwargs: Dict[str, Any],
) -> StageExecutionResult:
    name = stage.name
    tasks = stage.tasks or []
    if not tasks:
        return StageExecutionResult(stage_name=name, tasks_results={}, completed=True, error="No tasks found in stage")
    results: Dict[str, Any] = {}
    for task in tasks:
        tres = await execute_task(ctx, task, inputs, progress_callback, kwargs)
        results[task.name] = tres.result
        ctx.response_store.add(name, task.name, tres)
    return StageExecutionResult(stage_name=name, tasks_results=results, completed=True)

async def execute_stage_parallel(
    ctx: EngineContext,
    stage: WorkflowStage,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback],
    kwargs: Dict[str, Any],
) -> StageExecutionResult:
    name = stage.name
    tasks = stage.tasks or []
    if not tasks:
        return StageExecutionResult(stage_name=name, tasks_results={}, completed=True, error="No tasks found in stage")
    coros = [execute_task(ctx, task, inputs, progress_callback, kwargs) for task in tasks]
    outs = await asyncio.gather(*coros)
    results: Dict[str, Any] = {}
    for tres in outs:
        key = tres.task_name
        val = tres.result
        if not isinstance(val, dict):
            val = {"result": val}
        safe = _safe_serialize(val)
        results[key] = safe
        ctx.response_store.add(name, key, TaskExecutionResult(key, safe, completed=True))
    return StageExecutionResult(stage_name=name, tasks_results=results, completed=True)

async def execute_stage_handoff(
    ctx: EngineContext,
    stage: WorkflowStage,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback],
    kwargs: Dict[str, Any],
) -> StageExecutionResult:
    
    # Start by setting up the handoff stage
    name = stage.name
    tasks = stage.tasks or []
    if not tasks:
        logger.warning(f"Stage {name} has no handoff tasks defined, stopping execution")
        return StageExecutionResult(stage_name=name, tasks_results={}, completed=True, error="No handoff tasks found in stage")
    results: Dict[str, Any] = {}
    agent_list: list[Any] = [task.initialized_agent for task in tasks]
    logger.info(f"Tasks: {[task.name for task in tasks]}")
    handoff_agent = ctx.Agent(name="Handoff Agent", instructions=(stage.description), handoffs=agent_list)
    logger.info(f"Handoff agent intialized with config: {handoff_agent.handoffs}")

    # After setup, execute the handoff and send it to logs
    try:
        output_runner = await ctx.Runner.run(handoff_agent, inputs.user_query, hooks=ctx.hooks)
        key = stage.name
        val = output_runner.final_output
        if not isinstance(val, dict):
            val = {"result": val}
        safe = _safe_serialize(val)
        results[key] = safe
        ctx.response_store.add(stage.name, str(stage.tasks), TaskExecutionResult(key, safe, completed=True))
        logger.info(f"Output from handoff agent: {output_runner}")
        logger.info(f"Results after handoff: {results}")
    except Exception as e:
        logger.error(f"Error occurred during handoff execution: {e}")
        return StageExecutionResult(stage_name=stage.name, tasks_results=results, completed=False, error=str(e))

    #Callbacks
    if progress_callback:
        await progress_callback.on_stage_complete(name, {"tasks_results": results})
    if not results:
        logger.warning(f"Stage {name} completed with no results")
        return StageExecutionResult(stage_name=name, tasks_results=results, completed=True, error="No results from tasks")
    logger.info(f"Stage {name} completed with results: {list(results.keys())}")
    if progress_callback:
        await progress_callback.on_stage_complete(name, {"tasks_results": results})

    # Return a StageExecutionResult with the results
    return StageExecutionResult(stage_name=stage.name, tasks_results=results, completed=True)

async def execute_task(
    ctx: EngineContext,
    task: WorkflowTask,
    inputs: WorkflowInput,
    progress_callback: Optional[ProgressCallback],
    kwargs: Dict[str, Any],
) -> TaskExecutionResult:

    async def _fail(task_name_: str, e_: Exception) -> TaskExecutionResult:
        tr_ = TaskExecutionResult(task_name_, {"error": str(e_)}, completed=False, error=str(e_))
        if progress_callback:
            await progress_callback.on_task_fail(task_name_, str(e_), AgentOutput(agent=task.agent.id, output=tr_.result))
        return tr_

    task_name = task.name
    if progress_callback:
        await progress_callback.on_task_start(task_name, task)
    processed = task.process_inputs(ctx.response_store, inputs)
    logger.info(f"Executing task {task_name} with inputs {list(processed.keys())}")
    agent = getattr(task, "initialized_agent", None)
    if agent is None:
        return await _fail(task_name, ValueError(f"Task {task_name} has no initialized agent"))
    enforce_structured = False
    if task.provider and hasattr(task.provider, 'enforce_structured_output'):
        enforce_structured = task.provider.enforce_structured_output
        
    prompt = _create_prompt(
        ctx,
        task,
        inputs,
        processed,
        append_output_structure=not enforce_structured,
    )

    # Create a task-level trace for LLM observability
    await ctx.llm_tracer.start_task_trace(task_name=task_name,
                                    system_prompt=task.agent.system_prompt,
                                    prompt=prompt)

    try:
        runner = kwargs.get("runner")
        if not runner or not isinstance(runner, ctx.Runner):
            runner = ctx.Runner()
        result = await runner.run(agent, input=prompt, hooks=ctx.hooks)
        raw = getattr(result, "final_output", str(result))
        out = task.process_output(raw)
        
        # Log the LLM call
        model_name = task.provider.model if task.provider else "unknown"
        
        # Trace the LLM call
        await ctx.llm_tracer.trace_llm_call(
            name=task_name,
            model=model_name,
            system_prompt=task.agent.system_prompt,
            prompt=prompt,
            response=raw,
        )

        # Extract usage metrics from the context_wrapper into a dictionary
        usage = result.context_wrapper.usage
        metadata_dict = {
            "request_data": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens
            },
            "tags": {
                "requests": str(usage.requests)
            }
        }
        
        # End the task trace with metadata
        await ctx.llm_tracer.end_task_trace(
            status=TraceStatus.SUCCESS,
            system_prompt=task.agent.system_prompt,
            prompt=prompt,
            response=raw,
            metadata_dict=metadata_dict
        )
        
        tr = TaskExecutionResult(task_name, out, completed=True, structured_output_enforced=enforce_structured)
        ao = AgentOutput(agent=task_name, output=out)
        if progress_callback:
            await progress_callback.on_task_complete(task_name, {"outputs": out}, ao)
        return tr
    except Exception as e:
        # Log error in the LLM observability

        # Create error metadata dictionary
        metadata_dict = {
            "request_data": {
                "input_tokens": 0,  # We don't have metrics in case of error
                "output_tokens": 0,
                "total_tokens": 0
            },
            "tags": {
                "error": str(e),
                "error_type": type(e).__name__
            }
        }
        
        # End the task trace with error and metadata
        await ctx.llm_tracer.end_task_trace(
            status=TraceStatus.FAILED,
            error=e,
            system_prompt=task.agent.system_prompt,
            prompt=prompt,
            response=None,
            metadata_dict=metadata_dict
        )
        logger.error(f"Error executing task {task_name}: {e}")
        return await _fail(task_name, e)


# ----------------------------------------------------------------------------
# 5) Thin wrapper class to preserve the original public API
# ----------------------------------------------------------------------------

class OpenAIExecutionEngine(ExecutionEngine):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        mcp_registry: MCPServerRegistry,
        llm_tracer: LLMTracer,
    ):
        self._ctx = init_context(tool_registry, mcp_registry, llm_tracer)

    async def initialize_workflow(
        self,
        workflow: Workflow,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow:
        return await initialize_workflow(self._ctx, workflow, progress_callback)

    async def execute_workflow(
        self,
        workflow: Workflow,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        return await execute_workflow(
            self._ctx,
            workflow,
            inputs,
            progress_callback,
            **(kwargs or {}),
        )

    async def execute_task(
        self,
        task: WorkflowTask,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutionResult:
        """
        Delegate to the functional execute_task implementation.
        """
        return await execute_task(
            self._ctx,
            task,
            inputs,
            progress_callback,
            kwargs or {},
        )

    async def execute_stage(
        self,
        stage: WorkflowStage,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> StageExecutionResult:
        """
        Delegate to the functional execute_stage implementation.
        """
        return await execute_stage(
            self._ctx,
            stage,
            inputs,
            progress_callback,
            kwargs or {},
        )

# ----------------------------------------------------------------------------
# 6) Register the engine as before
# ----------------------------------------------------------------------------
ExecutionEngineFactory.register_engine("openai", OpenAIExecutionEngine)