import React, { useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Paper, Typography, Chip, IconButton, Tooltip } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import { WorkflowTask } from '../WorkflowContext';
import TaskDetailsDialog from '../dialogs/TaskDetailsDialog';

type TaskNodeData = {
  label: string;
  task: WorkflowTask;
  stageId: string;
};

const TaskNode: React.FC<NodeProps<TaskNodeData>> = ({ data, isConnectable }) => {
  const { label, task } = data;
  const [isDetailsOpen, setDetailsOpen] = useState(false);
  
  const handleOpenDetails = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDetailsOpen(true);
  };

  const handleCloseDetails = () => {
    setDetailsOpen(false);
  };
  
  return (
    <>
      <Paper
        sx={{
          padding: 2,
          width: 220,
          border: '1px solid #ddd',
          backgroundColor: '#fff',
          borderRadius: 2,
          position: 'relative',
        }}
      >
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: '#555' }}
          isConnectable={isConnectable}
        />
        
        <Box sx={{ mb: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 'bold',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: '80%'
              }}
            >
              {label}
            </Typography>
            <Tooltip title="Task Settings">
              <IconButton size="small" onClick={handleOpenDetails}>
                <SettingsIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              display: 'block',
              mb: 1,
            }}
          >
            {task.description || 'No description'}
          </Typography>
          
          <Chip
            label={`Agent: ${task.agent}`}
            size="small"
            sx={{
              backgroundColor: '#e8f5e9',
              color: '#2e7d32',
              fontSize: '0.7rem',
              height: 20
            }}
          />
        </Box>
        
        <Handle
          type="source"
          position={Position.Right}
          id="a"
          style={{ top: 20, background: '#555' }}
          isConnectable={isConnectable}
        />
        
        <Handle
          type="source"
          position={Position.Bottom}
          id="b"
          style={{ background: '#555' }}
          isConnectable={isConnectable}
        />
      </Paper>
      
      <TaskDetailsDialog
        open={isDetailsOpen}
        onClose={handleCloseDetails}
        task={task}
      />
    </>
  );
};

export default TaskNode;