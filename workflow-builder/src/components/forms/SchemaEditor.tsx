import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  IconButton,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';

interface SchemaField {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default?: string;
}

interface SchemaEditorProps {
  schema: Record<string, any>;
  onChange: (schema: Record<string, any>) => void;
  schemaType: 'input' | 'output';
}

const SchemaEditor: React.FC<SchemaEditorProps> = ({ schema, onChange, schemaType }) => {
  const [fields, setFields] = useState<SchemaField[]>(
    Object.keys(schema).map((key) => ({
      name: key,
      type: schema[key].type || 'str',
      description: schema[key].description || '',
      required: schema[key].required !== false, // Default to true
      default: schema[key].default || '',
    }))
  );

  const [newField, setNewField] = useState<SchemaField>({
    name: '',
    type: 'str',
    description: '',
    required: true,
    default: '',
  });

  const typeOptions = [
    { value: 'str', label: 'String' },
    { value: 'int', label: 'Integer' },
    { value: 'float', label: 'Float' },
    { value: 'bool', label: 'Boolean' },
    { value: 'list', label: 'List' },
    { value: 'dict', label: 'Dictionary' },
  ];

  const updateFields = (updatedFields: SchemaField[]) => {
    setFields(updatedFields);
    
    // Convert fields to schema format
    const newSchema: Record<string, any> = {};
    updatedFields.forEach((field) => {
      newSchema[field.name] = {
        type: field.type,
        description: field.description,
        required: field.required,
      };
      
      if (field.default) {
        newSchema[field.name].default = field.default;
      }
    });
    
    onChange(newSchema);
  };

  const handleFieldChange = (index: number, field: Partial<SchemaField>) => {
    const updatedFields = [...fields];
    updatedFields[index] = { ...updatedFields[index], ...field };
    updateFields(updatedFields);
  };

  const handleNewFieldChange = (field: Partial<SchemaField>) => {
    setNewField({ ...newField, ...field });
  };

  const addField = () => {
    if (newField.name) {
      updateFields([...fields, newField]);
      setNewField({
        name: '',
        type: 'str',
        description: '',
        required: true,
        default: '',
      });
    }
  };

  const removeField = (index: number) => {
    const updatedFields = [...fields];
    updatedFields.splice(index, 1);
    updateFields(updatedFields);
  };

  return (
    <Box>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Field Name</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Description</TableCell>
            <TableCell>Required</TableCell>
            <TableCell>Default</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {fields.map((field, index) => (
            <TableRow key={index}>
              <TableCell>
                <TextField
                  size="small"
                  value={field.name}
                  onChange={(e) => handleFieldChange(index, { name: e.target.value })}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <FormControl fullWidth size="small">
                  <Select
                    value={field.type}
                    onChange={(e) => handleFieldChange(index, { type: e.target.value })}
                  >
                    {typeOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </TableCell>
              <TableCell>
                <TextField
                  size="small"
                  value={field.description}
                  onChange={(e) => handleFieldChange(index, { description: e.target.value })}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={field.required}
                      onChange={(e) => handleFieldChange(index, { required: e.target.checked })}
                      size="small"
                    />
                  }
                  label=""
                />
              </TableCell>
              <TableCell>
                <TextField
                  size="small"
                  value={field.default || ''}
                  onChange={(e) => handleFieldChange(index, { default: e.target.value })}
                  fullWidth
                  disabled={field.required}
                />
              </TableCell>
              <TableCell>
                <IconButton size="small" color="error" onClick={() => removeField(index)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
          
          {/* Add new field row */}
          <TableRow>
            <TableCell>
              <TextField
                size="small"
                value={newField.name}
                onChange={(e) => handleNewFieldChange({ name: e.target.value })}
                placeholder="New field name"
                fullWidth
              />
            </TableCell>
            <TableCell>
              <FormControl fullWidth size="small">
                <Select
                  value={newField.type}
                  onChange={(e) => handleNewFieldChange({ type: e.target.value as string })}
                >
                  {typeOptions.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </TableCell>
            <TableCell>
              <TextField
                size="small"
                value={newField.description}
                onChange={(e) => handleNewFieldChange({ description: e.target.value })}
                placeholder="Description"
                fullWidth
              />
            </TableCell>
            <TableCell>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newField.required}
                    onChange={(e) => handleNewFieldChange({ required: e.target.checked })}
                    size="small"
                  />
                }
                label=""
              />
            </TableCell>
            <TableCell>
              <TextField
                size="small"
                value={newField.default || ''}
                onChange={(e) => handleNewFieldChange({ default: e.target.value })}
                placeholder="Default value"
                fullWidth
                disabled={newField.required}
              />
            </TableCell>
            <TableCell>
              <IconButton 
                size="small" 
                color="primary" 
                onClick={addField}
                disabled={!newField.name}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </Box>
  );
};

export default SchemaEditor;