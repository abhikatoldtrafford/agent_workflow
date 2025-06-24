import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tab,
  Tabs,
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  IconButton,
  Tooltip,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import { useWorkflow, WorkflowTask } from '../WorkflowContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`task-tabpanel-${index}`}
      aria-labelledby={`task-tab-${index}`}
      {...other}
      style={{ padding: '16px 0' }}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

interface TaskDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  task: WorkflowTask;
}

const TaskDetailsDialog: React.FC<TaskDetailsDialogProps> = ({
  open,
  onClose,
  task,
}) => {
  const { updateTask, getAgentById } = useWorkflow();
  const [tabIndex, setTabIndex] = useState(0);
  const [taskData, setTaskData] = useState<WorkflowTask>(task);
  const [newInputKey, setNewInputKey] = useState('');
  const [newInputValue, setNewInputValue] = useState('');
  const [newOutputKey, setNewOutputKey] = useState('');
  const [newOutputValue, setNewOutputValue] = useState('');
  
  // Get agent information for input/output schema
  const agent = getAgentById(task.agent);
  const inputSchema = agent?.input_schema || {};
  const outputSchema = agent?.output_schema || {};

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const handleTaskNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTaskData({
      ...taskData,
      name: e.target.value,
    });
  };

  const handleTaskDescriptionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTaskData({
      ...taskData,
      description: e.target.value,
    });
  };

  const addInputMapping = () => {
    if (newInputKey && newInputValue) {
      const updatedInputs = {
        ...taskData.inputs,
        [newInputKey]: newInputValue,
      };
      
      setTaskData({
        ...taskData,
        inputs: updatedInputs,
      });
      
      setNewInputKey('');
      setNewInputValue('');
    }
  };

  const removeInputMapping = (key: string) => {
    const updatedInputs = { ...taskData.inputs };
    delete updatedInputs[key];
    
    setTaskData({
      ...taskData,
      inputs: updatedInputs,
    });
  };

  const addOutputMapping = () => {
    if (newOutputKey && newOutputValue) {
      const updatedOutputs = {
        ...taskData.outputs,
        [newOutputKey]: newOutputValue,
      };
      
      setTaskData({
        ...taskData,
        outputs: updatedOutputs,
      });
      
      setNewOutputKey('');
      setNewOutputValue('');
    }
  };

  const removeOutputMapping = (key: string) => {
    const updatedOutputs = { ...taskData.outputs };
    delete updatedOutputs[key];
    
    setTaskData({
      ...taskData,
      outputs: updatedOutputs,
    });
  };

  const handleSave = () => {
    updateTask(task.id, taskData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Task Details: {task.name}</DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 3 }}>
          <TextField
            label="Task Name"
            fullWidth
            margin="normal"
            value={taskData.name}
            onChange={handleTaskNameChange}
          />
          <TextField
            label="Description"
            fullWidth
            margin="normal"
            value={taskData.description}
            onChange={handleTaskDescriptionChange}
            multiline
            rows={2}
          />
        </Box>

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabIndex} onChange={handleTabChange}>
            <Tab label="Inputs" />
            <Tab label="Outputs" />
          </Tabs>
        </Box>

        <TabPanel value={tabIndex} index={0}>
          <Typography variant="subtitle2" gutterBottom>
            Define input mappings for this task:
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Available input parameters from agent schema:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
              {Object.keys(inputSchema).length > 0 ? (
                Object.keys(inputSchema).map(key => (
                  <Box 
                    key={key}
                    sx={{
                      px: 1,
                      py: 0.5,
                      bgcolor: 'info.50',
                      borderRadius: 1,
                      fontSize: '0.75rem'
                    }}
                  >
                    {key}
                  </Box>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No input parameters defined in agent schema
                </Typography>
              )}
            </Box>
          </Box>
          
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell>Value / Source</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(taskData.inputs).length > 0 ? (
                Object.keys(taskData.inputs).map((key) => (
                  <TableRow key={key}>
                    <TableCell>{key}</TableCell>
                    <TableCell>{taskData.inputs[key]}</TableCell>
                    <TableCell align="right">
                      <IconButton size="small" onClick={() => removeInputMapping(key)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={3} align="center">
                    No input mappings defined
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <Box sx={{ display: 'flex', mt: 2, alignItems: 'flex-start' }}>
            <TextField
              label="Parameter"
              size="small"
              value={newInputKey}
              onChange={(e) => setNewInputKey(e.target.value)}
              sx={{ mr: 1, flex: 1 }}
            />
            <TextField
              label="Value/Source"
              size="small"
              value={newInputValue}
              onChange={(e) => setNewInputValue(e.target.value)}
              helperText="Use ${workflow.inputs.key} to reference workflow inputs"
              sx={{ mr: 1, flex: 2 }}
            />
            <Tooltip title="Add Input Mapping">
              <IconButton 
                color="primary"
                onClick={addInputMapping}
                disabled={!newInputKey || !newInputValue}
              >
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </TabPanel>

        <TabPanel value={tabIndex} index={1}>
          <Typography variant="subtitle2" gutterBottom>
            Define output mappings for this task:
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Available output parameters from agent schema:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
              {Object.keys(outputSchema).length > 0 ? (
                Object.keys(outputSchema).map(key => (
                  <Box 
                    key={key}
                    sx={{
                      px: 1,
                      py: 0.5,
                      bgcolor: 'success.50',
                      borderRadius: 1,
                      fontSize: '0.75rem'
                    }}
                  >
                    {key}
                  </Box>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No output parameters defined in agent schema
                </Typography>
              )}
            </Box>
          </Box>
          
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Workflow Output</TableCell>
                <TableCell>Agent Output Source</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(taskData.outputs).length > 0 ? (
                Object.keys(taskData.outputs).map((key) => (
                  <TableRow key={key}>
                    <TableCell>{key}</TableCell>
                    <TableCell>{taskData.outputs[key]}</TableCell>
                    <TableCell align="right">
                      <IconButton size="small" onClick={() => removeOutputMapping(key)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={3} align="center">
                    No output mappings defined
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <Box sx={{ display: 'flex', mt: 2, alignItems: 'flex-start' }}>
            <TextField
              label="Output Name"
              size="small"
              value={newOutputKey}
              onChange={(e) => setNewOutputKey(e.target.value)}
              sx={{ mr: 1, flex: 1 }}
            />
            <TextField
              label="Agent Output Source"
              size="small"
              value={newOutputValue}
              onChange={(e) => setNewOutputValue(e.target.value)}
              helperText="Use ${agent.output_schema.key} to reference agent outputs"
              sx={{ mr: 1, flex: 2 }}
            />
            <Tooltip title="Add Output Mapping">
              <IconButton 
                color="primary"
                onClick={addOutputMapping}
                disabled={!newOutputKey || !newOutputValue}
              >
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </TabPanel>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" color="primary">
          Save Changes
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TaskDetailsDialog;