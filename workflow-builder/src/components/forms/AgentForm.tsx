import React, { useState, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Divider,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import { v4 as uuidv4 } from 'uuid';
import { Agent } from '../WorkflowContext';
import SchemaEditor from './SchemaEditor';
import ToolEditor from './ToolEditor';

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
      id={`agent-tabpanel-${index}`}
      aria-labelledby={`agent-tab-${index}`}
      {...other}
      style={{ padding: '16px 0' }}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

interface AgentFormProps {
  initialAgent: Agent | null;
  onSave: (agent: Agent) => void;
  onCancel: () => void;
  isCreating: boolean;
}

const AgentForm: React.FC<AgentFormProps> = ({
  initialAgent,
  onSave,
  onCancel,
  isCreating,
}) => {
  const defaultAgent: Agent = {
    id: uuidv4(),
    name: '',
    description: '',
    agent_type: 'LLMAgent',
    system_prompt: '',
    user_prompt: '',
    tools: [],
    input_schema: {},
    output_schema: {},
  };

  const [agent, setAgent] = useState<Agent>(initialAgent || defaultAgent);
  const [tabIndex, setTabIndex] = useState(0);
  const [isToolDialogOpen, setToolDialogOpen] = useState(false);

  useEffect(() => {
    if (initialAgent) {
      setAgent(initialAgent);
    } else {
      setAgent(defaultAgent);
    }
  }, [initialAgent]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const handleInputChange = (field: keyof Agent, value: any) => {
    setAgent({
      ...agent,
      [field]: value,
    });
  };

  const handleSystemPromptChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleInputChange('system_prompt', e.target.value);
  };

  const handleUserPromptChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleInputChange('user_prompt', e.target.value);
  };

  const updateInputSchema = (schema: Record<string, any>) => {
    setAgent({
      ...agent,
      input_schema: schema,
    });
  };

  const updateOutputSchema = (schema: Record<string, any>) => {
    setAgent({
      ...agent,
      output_schema: schema,
    });
  };

  const addTool = (tool: any) => {
    setAgent({
      ...agent,
      tools: [...(agent.tools || []), tool],
    });
    setToolDialogOpen(false);
  };

  const removeTool = (index: number) => {
    const updatedTools = [...(agent.tools || [])];
    updatedTools.splice(index, 1);
    setAgent({
      ...agent,
      tools: updatedTools,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(agent);
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Typography variant="h6">
        {isCreating ? 'Create Agent' : 'Edit Agent'}
      </Typography>
      <Divider sx={{ my: 2 }} />

      <Box sx={{ mb: 3 }}>
        <TextField
          label="Agent Name"
          fullWidth
          margin="normal"
          value={agent.name}
          onChange={(e) => handleInputChange('name', e.target.value)}
          required
        />
        <TextField
          label="Description"
          fullWidth
          margin="normal"
          value={agent.description}
          onChange={(e) => handleInputChange('description', e.target.value)}
          multiline
          rows={2}
        />
        <TextField
          label="Agent Type"
          fullWidth
          margin="normal"
          value={agent.agent_type}
          onChange={(e) => handleInputChange('agent_type', e.target.value)}
          required
        />
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabIndex} onChange={handleTabChange}>
          <Tab label="Prompts" />
          <Tab label="Tools" />
          <Tab label="Schema" />
        </Tabs>
      </Box>

      <TabPanel value={tabIndex} index={0}>
        <Typography variant="subtitle2" gutterBottom>
          Define the agent prompts:
        </Typography>

        <TextField
          label="System Prompt"
          fullWidth
          margin="normal"
          value={agent.system_prompt || ''}
          onChange={handleSystemPromptChange}
          multiline
          rows={6}
          placeholder="You are an expert assistant that helps with..."
        />

        <TextField
          label="User Prompt Template"
          fullWidth
          margin="normal"
          value={agent.user_prompt || ''}
          onChange={handleUserPromptChange}
          multiline
          rows={6}
          placeholder="Use ${placeholders} for dynamic content..."
          helperText="Use ${variable} syntax for dynamic variables"
        />
      </TabPanel>

      <TabPanel value={tabIndex} index={1}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="subtitle2">
            Configure tools for this agent:
          </Typography>
          
          <Button
            variant="outlined"
            size="small"
            startIcon={<AddIcon />}
            onClick={() => setToolDialogOpen(true)}
          >
            Add Tool
          </Button>
        </Box>

        {agent.tools && agent.tools.length > 0 ? (
          agent.tools.map((tool, index) => (
            <Accordion key={index} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', pr: 2 }}>
                  <Typography>{tool.name}</Typography>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeTool(index);
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="caption" display="block" gutterBottom>
                  {tool.description}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontWeight: 'bold' }}>
                  Parameters:
                </Typography>
                <Box sx={{ ml: 2 }}>
                  {tool.parameters && tool.parameters.properties ? (
                    Object.entries(tool.parameters.properties).map(([paramName, paramConfig]: [string, any]) => (
                      <Box key={paramName} sx={{ mb: 1 }}>
                        <Typography variant="caption" component="div">
                          <strong>{paramName}</strong>
                          {tool.parameters.required && tool.parameters.required.includes(paramName) && (
                            <Typography component="span" sx={{ color: 'error.main', ml: 0.5 }}>*</Typography>
                          )}
                        </Typography>
                        <Typography variant="caption" component="div" color="text.secondary">
                          Type: {paramConfig.type}
                        </Typography>
                        <Typography variant="caption" component="div" color="text.secondary">
                          {paramConfig.description || 'No description'}
                        </Typography>
                      </Box>
                    ))
                  ) : (
                    <Typography variant="caption">No parameters defined</Typography>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          ))
        ) : (
          <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
            No tools configured for this agent
          </Typography>
        )}

        <Dialog open={isToolDialogOpen} onClose={() => setToolDialogOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle>Add Tool</DialogTitle>
          <DialogContent>
            <ToolEditor onSave={addTool} />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setToolDialogOpen(false)}>Cancel</Button>
          </DialogActions>
        </Dialog>
      </TabPanel>

      <TabPanel value={tabIndex} index={2}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle2" gutterBottom>
            Input Schema:
          </Typography>
          <SchemaEditor 
            schema={agent.input_schema || {}} 
            onChange={updateInputSchema}
            schemaType="input" 
          />
        </Box>
        
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Output Schema:
          </Typography>
          <SchemaEditor 
            schema={agent.output_schema || {}} 
            onChange={updateOutputSchema}
            schemaType="output" 
          />
        </Box>
      </TabPanel>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button onClick={onCancel} sx={{ mr: 1 }}>
          Cancel
        </Button>
        <Button type="submit" variant="contained" color="primary">
          {isCreating ? 'Create Agent' : 'Save Changes'}
        </Button>
      </Box>
    </Box>
  );
};

export default AgentForm;