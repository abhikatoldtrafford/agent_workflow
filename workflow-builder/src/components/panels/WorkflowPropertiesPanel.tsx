import React, { useState } from 'react';
import { 
  Paper, 
  Typography, 
  TextField, 
  Button,
  Box,
  Divider
} from '@mui/material';
import { useWorkflow } from '../WorkflowContext';

const WorkflowPropertiesPanel: React.FC = () => {
  const { workflow, updateWorkflow } = useWorkflow();
  
  const [name, setName] = useState(workflow.name);
  const [description, setDescription] = useState(workflow.description);
  const [version, setVersion] = useState(workflow.version);
  
  const handleSave = () => {
    updateWorkflow({
      name,
      description,
      version
    });
  };
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        width: 300,
        bgcolor: 'rgba(255, 255, 255, 0.95)'
      }}
    >
      <Typography variant="h6" gutterBottom>
        Workflow Properties
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <TextField
        label="Workflow Name"
        fullWidth
        size="small"
        margin="dense"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      
      <TextField
        label="Description"
        fullWidth
        size="small"
        margin="dense"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        multiline
        rows={2}
      />
      
      <TextField
        label="Version"
        fullWidth
        size="small"
        margin="dense"
        value={version}
        onChange={(e) => setVersion(e.target.value)}
      />
      
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          variant="contained" 
          size="small" 
          onClick={handleSave}
        >
          Apply Changes
        </Button>
      </Box>
    </Paper>
  );
};

export default WorkflowPropertiesPanel;