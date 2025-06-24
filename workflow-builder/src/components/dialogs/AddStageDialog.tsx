import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
} from '@mui/material';
import { v4 as uuidv4 } from 'uuid';
import { useWorkflow, WorkflowStage } from '../WorkflowContext';

interface AddStageDialogProps {
  open: boolean;
  onClose: () => void;
}

const AddStageDialog: React.FC<AddStageDialogProps> = ({ open, onClose }) => {
  const { addStage } = useWorkflow();
  const [stageName, setStageName] = useState('');
  const [stageDescription, setStageDescription] = useState('');
  const [executionType, setExecutionType] = useState<'sequential' | 'parallel'>('sequential');
  const [nameError, setNameError] = useState('');

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setStageName(e.target.value);
    if (e.target.value) {
      setNameError('');
    }
  };

  const handleDescriptionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setStageDescription(e.target.value);
  };

  const handleExecutionTypeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setExecutionType(e.target.value as 'sequential' | 'parallel');
  };

  const handleSave = () => {
    if (!stageName) {
      setNameError('Stage name is required');
      return;
    }

    const newStage: WorkflowStage = {
      id: uuidv4(),
      name: stageName,
      description: stageDescription,
      execution_type: executionType,
    };

    addStage(newStage);
    handleClose();
  };

  const handleClose = () => {
    setStageName('');
    setStageDescription('');
    setExecutionType('sequential');
    setNameError('');
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Add Workflow Stage</DialogTitle>
      <DialogContent>
        <TextField
          label="Stage Name"
          fullWidth
          margin="normal"
          value={stageName}
          onChange={handleNameChange}
          error={!!nameError}
          helperText={nameError}
          autoFocus
        />
        <TextField
          label="Description"
          fullWidth
          margin="normal"
          value={stageDescription}
          onChange={handleDescriptionChange}
          multiline
          rows={2}
        />
        <FormControl component="fieldset" margin="normal">
          <FormLabel component="legend">Execution Type</FormLabel>
          <RadioGroup
            row
            name="execution-type"
            value={executionType}
            onChange={handleExecutionTypeChange}
          >
            <FormControlLabel
              value="sequential"
              control={<Radio />}
              label="Sequential"
            />
            <FormControlLabel
              value="parallel"
              control={<Radio />}
              label="Parallel"
            />
          </RadioGroup>
        </FormControl>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" color="primary">
          Add Stage
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AddStageDialog;