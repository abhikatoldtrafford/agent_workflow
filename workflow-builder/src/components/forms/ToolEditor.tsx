import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
  FormControlLabel,
  Checkbox,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';

interface Parameter {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

interface Tool {
  name: string;
  description: string;
  type: string;
  parameters: {
    type: string;
    properties: Record<string, any>;
    required: string[];
  };
}

interface ToolEditorProps {
  onSave: (tool: Tool) => void;
}

const ToolEditor: React.FC<ToolEditorProps> = ({ onSave }) => {
  const [tool, setTool] = useState<Tool>({
    name: '',
    description: '',
    type: 'function',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  });

  const [parameters, setParameters] = useState<Parameter[]>([]);
  const [newParameter, setNewParameter] = useState<Parameter>({
    name: '',
    type: 'string',
    description: '',
    required: true,
  });

  const parameterTypes = [
    { value: 'string', label: 'String' },
    { value: 'number', label: 'Number' },
    { value: 'boolean', label: 'Boolean' },
    { value: 'array', label: 'Array' },
    { value: 'object', label: 'Object' },
  ];

  const handleToolChange = (field: keyof Tool, value: any) => {
    setTool({
      ...tool,
      [field]: value,
    });
  };

  const handleParameterChange = (index: number, field: keyof Parameter, value: any) => {
    const updatedParameters = [...parameters];
    updatedParameters[index] = {
      ...updatedParameters[index],
      [field]: value,
    };

    setParameters(updatedParameters);
    updateToolParameters(updatedParameters);
  };

  const handleNewParameterChange = (field: keyof Parameter, value: any) => {
    setNewParameter({
      ...newParameter,
      [field]: value,
    });
  };

  const addParameter = () => {
    if (newParameter.name) {
      const updatedParameters = [...parameters, newParameter];
      setParameters(updatedParameters);
      updateToolParameters(updatedParameters);
      
      setNewParameter({
        name: '',
        type: 'string',
        description: '',
        required: true,
      });
    }
  };

  const removeParameter = (index: number) => {
    const updatedParameters = [...parameters];
    updatedParameters.splice(index, 1);
    setParameters(updatedParameters);
    updateToolParameters(updatedParameters);
  };

  const updateToolParameters = (params: Parameter[]) => {
    const properties: Record<string, any> = {};
    const required: string[] = [];

    params.forEach((param) => {
      properties[param.name] = {
        type: param.type,
        description: param.description,
      };

      if (param.required) {
        required.push(param.name);
      }
    });

    setTool({
      ...tool,
      parameters: {
        type: 'object',
        properties,
        required,
      },
    });
  };

  const handleSave = () => {
    onSave(tool);
  };

  const isFormValid = () => {
    return tool.name && tool.description;
  };

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Tool Configuration
      </Typography>

      <Box sx={{ mb: 3 }}>
        <TextField
          label="Tool Name"
          fullWidth
          margin="normal"
          value={tool.name}
          onChange={(e) => handleToolChange('name', e.target.value)}
          required
        />

        <TextField
          label="Description"
          fullWidth
          margin="normal"
          value={tool.description}
          onChange={(e) => handleToolChange('description', e.target.value)}
          multiline
          rows={2}
          required
        />

        <FormControl fullWidth margin="normal">
          <InputLabel id="tool-type-label">Tool Type</InputLabel>
          <Select
            labelId="tool-type-label"
            value={tool.type}
            label="Tool Type"
            onChange={(e) => handleToolChange('type', e.target.value)}
          >
            <MenuItem value="function">Function</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Typography variant="subtitle2" gutterBottom>
        Parameters
      </Typography>

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Description</TableCell>
            <TableCell>Required</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {parameters.map((param, index) => (
            <TableRow key={index}>
              <TableCell>
                <TextField
                  size="small"
                  value={param.name}
                  onChange={(e) => handleParameterChange(index, 'name', e.target.value)}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <FormControl fullWidth size="small">
                  <Select
                    value={param.type}
                    onChange={(e) => handleParameterChange(index, 'type', e.target.value)}
                  >
                    {parameterTypes.map((type) => (
                      <MenuItem key={type.value} value={type.value}>
                        {type.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </TableCell>
              <TableCell>
                <TextField
                  size="small"
                  value={param.description}
                  onChange={(e) => handleParameterChange(index, 'description', e.target.value)}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={param.required}
                      onChange={(e) => handleParameterChange(index, 'required', e.target.checked)}
                      size="small"
                    />
                  }
                  label=""
                />
              </TableCell>
              <TableCell>
                <IconButton size="small" color="error" onClick={() => removeParameter(index)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}

          {/* Add new parameter row */}
          <TableRow>
            <TableCell>
              <TextField
                size="small"
                value={newParameter.name}
                onChange={(e) => handleNewParameterChange('name', e.target.value)}
                placeholder="Parameter name"
                fullWidth
              />
            </TableCell>
            <TableCell>
              <FormControl fullWidth size="small">
                <Select
                  value={newParameter.type}
                  onChange={(e) => handleNewParameterChange('type', e.target.value as string)}
                >
                  {parameterTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </TableCell>
            <TableCell>
              <TextField
                size="small"
                value={newParameter.description}
                onChange={(e) => handleNewParameterChange('description', e.target.value)}
                placeholder="Description"
                fullWidth
              />
            </TableCell>
            <TableCell>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newParameter.required}
                    onChange={(e) => handleNewParameterChange('required', e.target.checked)}
                    size="small"
                  />
                }
                label=""
              />
            </TableCell>
            <TableCell>
              <IconButton
                size="small"
                color="primary"
                onClick={addParameter}
                disabled={!newParameter.name}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSave}
          disabled={!isFormValid()}
        >
          Add Tool
        </Button>
      </Box>
    </Box>
  );
};

export default ToolEditor;