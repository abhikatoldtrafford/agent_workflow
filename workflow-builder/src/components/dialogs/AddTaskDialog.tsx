import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
} from '@mui/material';
import { v4 as uuidv4 } from 'uuid';
import { useWorkflow, WorkflowTask } from '../WorkflowContext';

interface AddTaskDialogProps {
  open: boolean;
  onClose: () => void;
  selectedStageId: string | null;
}

const AddTaskDialog: React.FC<AddTaskDialogProps> = ({ 
  open, 
  onClose, 
  selectedStageId 
}) => {
  const { stages, addTask, agents } = useWorkflow();
  
  const [taskName, setTaskName] = useState('');
  const [taskDescription, setTaskDescription] = useState('');
  const [stageId, setStageId] = useState('');
  const [agentId, setAgentId] = useState('');
  
  const [nameError, setNameError] = useState('');
  const [stageError, setStageError] = useState('');
  const [agentError, setAgentError] = useState('');

  // Set selected stage when provided from parent
  useEffect(() => {
    if (selectedStageId) {
      setStageId(selectedStageId);
      setStageError('');
    }
  }, [selectedStageId]);

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTaskName(e.target.value);
    if (e.target.value) {
      setNameError('');
    }
  };

  const handleDescriptionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTaskDescription(e.target.value);
  };

  const handleStageChange = (e: React.ChangeEvent<{ value: unknown }>) => {
    setStageId(e.target.value as string);
    setStageError('');
  };

  const handleAgentChange = (e: React.ChangeEvent<{ value: unknown }>) => {
    setAgentId(e.target.value as string);
    setAgentError('');
  };

  const handleSave = () => {
    // Validate form
    let isValid = true;

    if (!taskName) {
      setNameError('Task name is required');
      isValid = false;
    }

    if (!stageId) {
      setStageError('Stage is required');
      isValid = false;
    }

    if (!agentId) {
      setAgentError('Agent is required');
      isValid = false;
    }

    if (!isValid) {
      return;
    }

    const newTask: WorkflowTask = {
      id: uuidv4(),
      name: taskName,
      description: taskDescription,
      stageId: stageId,
      agent: agentId,
      inputs: {},
      outputs: {},
    };

    addTask(newTask);
    handleClose();
  };

  const handleClose = () => {
    setTaskName('');
    setTaskDescription('');
    setStageId(selectedStageId || '');
    setAgentId('');
    setNameError('');
    setStageError('');
    setAgentError('');
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Add Task</DialogTitle>
      <DialogContent>
        <TextField
          label="Task Name"
          fullWidth
          margin="normal"
          value={taskName}
          onChange={handleNameChange}
          error={!!nameError}
          helperText={nameError}
          autoFocus
        />
        
        <TextField
          label="Description"
          fullWidth
          margin="normal"
          value={taskDescription}
          onChange={handleDescriptionChange}
          multiline
          rows={2}
        />
        
        <FormControl fullWidth margin="normal" error={!!stageError}>
          <InputLabel id="stage-select-label">Stage</InputLabel>
          <Select
            labelId="stage-select-label"
            id="stage-select"
            value={stageId}
            onChange={handleStageChange}
            label="Stage"
            disabled={!!selectedStageId}
          >
            {stages.map((stage) => (
              <MenuItem key={stage.id} value={stage.id}>
                {stage.name}
              </MenuItem>
            ))}
          </Select>
          {stageError && <FormHelperText>{stageError}</FormHelperText>}
        </FormControl>
        
        <FormControl fullWidth margin="normal" error={!!agentError}>
          <InputLabel id="agent-select-label">Agent</InputLabel>
          <Select
            labelId="agent-select-label"
            id="agent-select"
            value={agentId}
            onChange={handleAgentChange}
            label="Agent"
          >
            {agents.map((agent) => (
              <MenuItem key={agent.id} value={agent.id}>
                {agent.name}
              </MenuItem>
            ))}
          </Select>
          {agentError && <FormHelperText>{agentError}</FormHelperText>}
        </FormControl>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button 
          onClick={handleSave} 
          variant="contained" 
          color="primary"
        >
          Add Task
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AddTaskDialog;