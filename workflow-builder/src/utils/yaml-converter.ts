import YAML from 'js-yaml';
import { Workflow, WorkflowStage, WorkflowTask, Agent } from '../components/WorkflowContext';

// Function to convert a workflow to YAML
export const workflowToYAML = (
  workflow: Workflow,
  stages: WorkflowStage[],
  tasks: WorkflowTask[]
): string => {
  // Create a deep copy to avoid modifying the original objects
  const workflowObj = {
    name: workflow.name,
    description: workflow.description,
    version: workflow.version,
    stages: stages.map(stage => {
      // Get tasks associated with this stage
      const stageTasks = tasks.filter(task => task.stageId === stage.id);
      
      return {
        name: stage.name,
        description: stage.description,
        execution_type: stage.execution_type,
        tasks: stageTasks.map(task => ({
          name: task.name,
          description: task.description,
          agent: {
            ref: task.agent // Reference to the agent file
          },
          inputs: task.inputs,
          outputs: task.outputs
        }))
      };
    })
  };
  
  return YAML.dump(workflowObj, {
    indent: 2,
    lineWidth: -1,
    quotingType: '"'
  });
};

// Function to convert an agent to YAML
export const agentToYAML = (agent: Agent): string => {
  // Create a copy of the agent without the id for YAML export
  const { id, ...agentWithoutId } = agent;
  
  return YAML.dump(agentWithoutId, {
    indent: 2,
    lineWidth: -1,
    quotingType: '"'
  });
};

// Function to parse YAML back to workflow objects
export const parseWorkflowYAML = (yamlString: string): {
  workflow: Partial<Workflow>;
  stages: WorkflowStage[];
  tasks: WorkflowTask[];
} => {
  try {
    const parsedWorkflow = YAML.load(yamlString) as any;
    
    const workflow: Partial<Workflow> = {
      name: parsedWorkflow.name,
      description: parsedWorkflow.description,
      version: parsedWorkflow.version,
    };
    
    const stages: WorkflowStage[] = [];
    const tasks: WorkflowTask[] = [];
    
    // Process stages and tasks
    if (parsedWorkflow.stages && Array.isArray(parsedWorkflow.stages)) {
      parsedWorkflow.stages.forEach((stageData: any, stageIndex: number) => {
        const stageId = `stage-${stageIndex}-${Date.now()}`;
        
        const stage: WorkflowStage = {
          id: stageId,
          name: stageData.name || `Stage ${stageIndex + 1}`,
          description: stageData.description || '',
          execution_type: stageData.execution_type || 'sequential',
        };
        
        stages.push(stage);
        
        // Process tasks in this stage
        if (stageData.tasks && Array.isArray(stageData.tasks)) {
          stageData.tasks.forEach((taskData: any, taskIndex: number) => {
            const taskId = `${stageIndex}-${taskIndex}-${Date.now()}`;
            
            const task: WorkflowTask = {
              id: taskId,
              name: taskData.name || `Task ${taskIndex + 1}`,
              description: taskData.description || '',
              stageId: stageId,
              agent: typeof taskData.agent === 'string' 
                ? taskData.agent 
                : (taskData.agent?.ref || 'unknown'),
              inputs: taskData.inputs || {},
              outputs: taskData.outputs || {},
            };
            
            tasks.push(task);
          });
        }
      });
    }
    
    return { workflow, stages, tasks };
  } catch (error) {
    console.error('Failed to parse workflow YAML:', error);
    throw error;
  }
};

// Function to parse YAML back to agent object
export const parseAgentYAML = (yamlString: string): Agent => {
  try {
    const parsedAgent = YAML.load(yamlString) as any;
    
    // Generate a new ID for the imported agent
    const importedAgent: Agent = {
      id: `agent-${Date.now()}`,
      name: parsedAgent.name || 'Imported Agent',
      description: parsedAgent.description || '',
      agent_type: parsedAgent.agent_type || 'LLMAgent',
      system_prompt: parsedAgent.system_prompt,
      user_prompt: parsedAgent.user_prompt,
      tools: parsedAgent.tools || [],
      input_schema: parsedAgent.input_schema || {},
      output_schema: parsedAgent.output_schema || {},
    };
    
    return importedAgent;
  } catch (error) {
    console.error('Failed to parse agent YAML:', error);
    throw error;
  }
};