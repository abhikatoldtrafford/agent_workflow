import React, { useState } from 'react';
import { Box, Grid, Paper, Button} from '@mui/material';
import ReactFlow, {
  Controls,
  Background,
  Panel,
  Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useWorkflow } from './WorkflowContext';
import StageNode from './nodes/StageNode';
import TaskNode from './nodes/TaskNode';
import AddStageDialog from './dialogs/AddStageDialog';
import AddTaskDialog from './dialogs/AddTaskDialog';
import WorkflowPropertiesPanel from './panels/WorkflowPropertiesPanel';

// Define custom node types
const nodeTypes = {
  stageNode: StageNode,
  taskNode: TaskNode,
};

const WorkflowEditor: React.FC = () => {
  const { 
    // workflow,
    stages, 
    // tasks,
    nodes, 
    edges, 
    onNodesChange, 
    onEdgesChange, 
    onConnect 
  } = useWorkflow();

  const [isStageDialogOpen, setStageDialogOpen] = useState(false);
  const [isTaskDialogOpen, setTaskDialogOpen] = useState(false);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [isPanelOpen, setPanelOpen] = useState(false);

  const handleAddStage = () => {
    setStageDialogOpen(true);
  };

  const handleAddTask = () => {
    setTaskDialogOpen(true);
  };

  const handleStageDialogClose = () => {
    setStageDialogOpen(false);
  };

  const handleTaskDialogClose = () => {
    setTaskDialogOpen(false);
    setSelectedStage(null);
  };

  const handleStageSelect = (stageId: string) => {
    setSelectedStage(stageId);
    setTaskDialogOpen(true);
  };

  const handleEdgeConnect = (params: Connection) => {
    // Validate the connection before adding it
    if (params.source && params.target) {
      // Example validation:
      // 1. Check if source output matches target input schema
      // 2. Prevent circular connections

      // If validation passes, add the connection
      onConnect(params);
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleAddStage}
          >
            Add Stage
          </Button>
        </Grid>
        <Grid item>
          <Button 
            variant="contained" 
            onClick={handleAddTask}
            disabled={stages.length === 0}
          >
            Add Task
          </Button>
        </Grid>
        <Grid item>
          <Button 
            variant="outlined" 
            onClick={() => setPanelOpen(!isPanelOpen)}
          >
            {isPanelOpen ? 'Hide Properties' : 'Show Properties'}
          </Button>
        </Grid>
      </Grid>

      <Paper sx={{ flexGrow: 1, position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={handleEdgeConnect}
          nodeTypes={nodeTypes}
          fitView
        >
          <Controls />
          <Background />
          
          {isPanelOpen && (
            <Panel position="top-right">
              <WorkflowPropertiesPanel />
            </Panel>
          )}
        </ReactFlow>
      </Paper>

      <AddStageDialog
        open={isStageDialogOpen}
        onClose={handleStageDialogClose}
      />

      <AddTaskDialog
        open={isTaskDialogOpen}
        onClose={handleTaskDialogClose}
        selectedStageId={selectedStage}
      />
    </Box>
  );
};

export default WorkflowEditor;