import React, { createContext, useState, ReactNode, useContext, useRef, useCallback } from 'react';
import { 
  Edge, 
  Node, 
  addEdge, 
  Connection,
  applyNodeChanges as applyNodeChangesRF,
  applyEdgeChanges as applyEdgeChangesRF,
  NodeChange,
  EdgeChange
} from 'reactflow';

// Define types for our workflow data
export type WorkflowStage = {
  id: string;
  name: string;
  description: string;
  execution_type: 'sequential' | 'parallel';
};

export type WorkflowTask = {
  id: string;
  name: string;
  description: string;
  agent: string;
  stageId: string;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
};

export type Agent = {
  id: string;
  name: string;
  description: string;
  agent_type: string;
  system_prompt?: string;
  user_prompt?: string;
  tools?: Array<{
    name: string;
    description: string;
    type: string;
    parameters: {
      type: string;
      properties: Record<string, any>;
      required: string[];
    };
  }>;
  input_schema?: Record<string, any>;
  output_schema?: Record<string, any>;
};

export type Workflow = {
  name: string;
  description: string;
  version: string;
  stages: WorkflowStage[];
};

type WorkflowContextType = {
  workflow: Workflow;
  agents: Agent[];
  stages: WorkflowStage[];
  tasks: WorkflowTask[];
  nodes: Node[];
  edges: Edge[];
  updateWorkflow: (workflow: Partial<Workflow>) => void;
  addStage: (stage: WorkflowStage) => void;
  updateStage: (stageId: string, data: Partial<WorkflowStage>) => void;
  deleteStage: (stageId: string) => void;
  addTask: (task: WorkflowTask) => void;
  updateTask: (taskId: string, data: Partial<WorkflowTask>) => void;
  deleteTask: (taskId: string) => void;
  addAgent: (agent: Agent) => void;
  updateAgent: (agentId: string, data: Partial<Agent>) => void;
  deleteAgent: (agentId: string) => void;
  onNodesChange: (changes: any) => void;
  onEdgesChange: (changes: any) => void;
  onConnect: (connection: Connection) => void;
  getAgentById: (id: string) => Agent | undefined;
  getTaskById: (id: string) => WorkflowTask | undefined;
  getStageById: (id: string) => WorkflowStage | undefined;
};

function useSynchronousState<T>(initial: T): [T, (value: T) => void, React.MutableRefObject<T>] {
  const [state, setState] = useState<T>(initial);
  const ref = useRef<T>(state);

  const setBoth = useCallback((value: T) => {
    ref.current = value;
    setState(value);
  }, []);

  return [state, setBoth, ref];
}

// Create the context with a default empty value
export const WorkflowContext = createContext<WorkflowContextType | undefined>(undefined);

// Context provider component
export const WorkflowProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [workflow, setWorkflow] = useState<Workflow>({
    name: 'New Workflow',
    description: 'A workflow created with the Workflow Builder',
    version: '1.0.0',
    stages: [],
  });

  const [agents, setAgents, agentsRef] = useSynchronousState<Agent[]>([]);
  const [stages, setStages, stagesRef] = useSynchronousState<WorkflowStage[]>([]);
  const [tasks, setTasks, tasksRef] = useSynchronousState<WorkflowTask[]>([]);
  const [nodes, setNodes, nodesRef] = useSynchronousState<Node[]>([]);
  const [edges, setEdges, edgesRef] = useSynchronousState<Edge[]>([]);

  const updateWorkflow = (data: Partial<Workflow>) => {
    setWorkflow(prev => ({ ...prev, ...data }));
  };

  const addStage = (stage: WorkflowStage) => {
    setStages([...stagesRef.current, stage]);
    
    // Add a node for the stage
    const newNode: Node = {
      id: `stage-${stage.id}`,
      type: 'stageNode',
      data: { label: stage.name, stage },
      position: { x: 100, y: stages.length * 150 },
    };

    setNodes([...nodesRef.current, newNode])
  };

  const updateStage = (stageId: string, data: Partial<WorkflowStage>) => {
    const updatedStages = stagesRef.current.map(stage =>
        stage.id === stageId ? { ...stage, ...data } : stage
    );
    setStages(updatedStages);

    const updatedNodes = nodesRef.current.map(node => {
      if (node.id === `stage-${stageId}`) {
        const updatedStage = { ...node.data.stage, ...data };
        return {
          ...node,
          data: {
            ...node.data,
            label: updatedStage.name,
            stage: updatedStage,
          },
        };
      }
      return node;
    });
    setNodes(updatedNodes);
  };

  const deleteStage = (stageId: string) => {
    // Delete the stage
    const filterStage = stagesRef.current.filter(stage => stage.id !== stageId);
    setStages(filterStage)
    
    // Delete all tasks in this stage
    const tasksToDelete = tasksRef.current.filter(task => task.stageId === stageId);
    const taskIds = tasksToDelete.map(task => task.id);

    const filterTask = tasksRef.current.filter(task => task.stageId !== stageId);
    setTasks(filterTask);
    
    // Delete corresponding nodes and edges
    const filterNode = nodesRef.current.filter(node =>
        node.id !== `stage-${stageId}` &&
        !taskIds.some(id => node.id === `task-${id}`)
    );
    setNodes(filterNode);

    const filterEdge = edgesRef.current.filter(edge =>
        !edge.source.startsWith(`stage-${stageId}`) &&
        !edge.target.startsWith(`stage-${stageId}`) &&
        !taskIds.some(id => 
            edge.source === `task-${id}` || 
            edge.target === `task-${id}`
        )
    );
    setEdges(filterEdge);
  };

  const addTask = (task: WorkflowTask) => {
    setTasks([...tasksRef.current, task]);
    
    // Find tasks in the same stage to determine position
    const stageTasks = tasksRef.current.filter(t => t.stageId === task.stageId);
    const stage = stagesRef.current.find(s => s.id === task.stageId);
    
    if (stage) {
      // Add a node for the task
      const stageNode = nodesRef.current.find(n => n.id === `stage-${stage.id}`);
      const stageX = stageNode ? stageNode.position.x : 100;
      const stageY = stageNode ? stageNode.position.y : 100;
      
      const newNode: Node = {
        id: `task-${task.id}`,
        type: 'taskNode',
        data: { label: task.name, task, stageId: stage.id },
        position: { 
          x: stageX + 250, 
          y: stageY + (stageTasks.length * 120)
        },
      };

      setNodes([...nodesRef.current, newNode]);

      // Connect the task to its stage
      const newEdge: Edge = {
        id: `edge-${stage.id}-${task.id}`,
        source: `stage-${stage.id}`,
        target: `task-${task.id}`,
        type: 'smoothstep',
      };

      setEdges([...edgesRef.current, newEdge]);
    }
  };

  const updateTask = (taskId: string, data: Partial<WorkflowTask>) => {
    const updatedTasks = tasksRef.current.map(task =>
        task.id === taskId ? { ...task, ...data } : task
    );
    setTasks(updatedTasks);

    const updatedNodes = nodesRef.current.map(node => {
      if (node.id === `task-${taskId}`) {
        const updatedTask = { ...node.data.task, ...data };
        return {
          ...node,
          data: {
            ...node.data,
            label: updatedTask.name,
            task: updatedTask,
          },
        };
      }
      return node;
    });
    setNodes(updatedNodes);
  };

  const deleteTask = (taskId: string) => {
    // Delete the task
    const filteredTasks = tasksRef.current.filter(task => task.id !== taskId);
    setTasks(filteredTasks);

    // Delete corresponding node and edges
    const filteredNodes = nodesRef.current.filter(node => node.id !== `task-${taskId}`);
    setNodes(filteredNodes);

    const filteredEdges = edgesRef.current.filter(edge =>
        edge.source !== `task-${taskId}` &&
        edge.target !== `task-${taskId}`
    );
    setEdges(filteredEdges);
  };

  const addAgent = (agent: Agent) => {
    setAgents([...agentsRef.current, agent]);
  };

  const updateAgent = (agentId: string, data: Partial<Agent>) => {
    const updatedAgents = agentsRef.current.map(agent =>
        agent.id === agentId ? { ...agent, ...data } : agent
    );
    setAgents(updatedAgents);
  };

  const deleteAgent = (agentId: string) => {
    const filteredAgents = agentsRef.current.filter(agent => agent.id !== agentId);
    setAgents(filteredAgents);
  };

  const onNodesChange = (changes: NodeChange[]) => {
    const updatedNodes = applyNodeChangesRF(changes, nodesRef.current);
    setNodes(updatedNodes);
  };

  const onEdgesChange = (changes: EdgeChange[]) => {
    const updatedEdges = applyEdgeChangesRF(changes, edgesRef.current);
    setEdges(updatedEdges);
  };

  const onConnect = (connection: Connection) => {
    const newEdges = addEdge({ ...connection, type: 'smoothstep' }, edgesRef.current);
    setEdges(newEdges);
  };

  // Helper functions to get items by ID
  const getAgentById = (id: string) => agentsRef.current.find(agent => agent.id === id);
  
  const getTaskById = (id: string) => tasksRef.current.find(task => task.id === id);
  
  const getStageById = (id: string) => stagesRef.current.find(stage => stage.id === id);

  return (
    <WorkflowContext.Provider
      value={{
        workflow,
        agents,
        stages,
        tasks,
        nodes,
        edges,
        updateWorkflow,
        addStage,
        updateStage,
        deleteStage,
        addTask,
        updateTask,
        deleteTask,
        addAgent,
        updateAgent,
        deleteAgent,
        onNodesChange,
        onEdgesChange,
        onConnect,
        getAgentById,
        getTaskById,
        getStageById,
      }}
    >
      {children}
    </WorkflowContext.Provider>
  );
};

// Custom hook to use the workflow context
export const useWorkflow = () => {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error('useWorkflow must be used within a WorkflowProvider');
  }
  return context;
};