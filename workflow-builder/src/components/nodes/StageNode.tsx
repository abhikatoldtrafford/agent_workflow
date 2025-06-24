import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Paper, Typography, Chip } from '@mui/material';
import { WorkflowStage } from '../WorkflowContext';

type StageNodeData = {
  label: string;
  stage: WorkflowStage;
};

const StageNode: React.FC<NodeProps<StageNodeData>> = ({ data, isConnectable }) => {
  const { label, stage } = data;
  
  return (
    <Paper
      sx={{
        padding: 2,
        width: 200,
        border: '1px solid #ddd',
        backgroundColor: '#f8f8f8',
        borderRadius: 2,
      }}
    >
      <Handle
        type="source"
        position={Position.Right}
        id="a"
        style={{ top: 20, background: '#555' }}
        isConnectable={isConnectable}
      />
      
      <Box sx={{ mb: 1 }}>
        <Typography
          variant="subtitle1"
          sx={{
            fontWeight: 'bold',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            display: 'block',
          }}
        >
          {stage.description || 'No description'}
        </Typography>
      </Box>
      
      <Chip
        label={stage.execution_type}
        size="small"
        sx={{
          backgroundColor: stage.execution_type === 'sequential' ? '#e3f2fd' : '#fff8e1',
          color: stage.execution_type === 'sequential' ? '#0d47a1' : '#e65100',
        }}
      />
      
      <Handle
        type="target"
        position={Position.Left}
        id="b"
        style={{ background: '#555' }}
        isConnectable={isConnectable}
      />
    </Paper>
  );
};

export default StageNode;