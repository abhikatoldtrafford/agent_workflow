import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Divider,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';

import { useWorkflow, Agent } from './WorkflowContext';
import AgentForm from './forms/AgentForm';

const AgentEditor: React.FC = () => {
  const { agents, addAgent, updateAgent, deleteAgent } = useWorkflow();
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateAgent = () => {
    setSelectedAgent(null);
    setIsCreating(true);
  };

  const handleEditAgent = (agent: Agent) => {
    setSelectedAgent(agent);
    setIsCreating(false);
  };

  const handleDeleteAgent = (agentId: string) => {
    if (window.confirm('Are you sure you want to delete this agent?')) {
      deleteAgent(agentId);
      if (selectedAgent?.id === agentId) {
        setSelectedAgent(null);
        setIsCreating(false);
      }
    }
  };

  const handleSaveAgent = (agent: Agent) => {
    if (isCreating) {
      addAgent(agent);
    } else {
      updateAgent(agent.id, agent);
    }
    setSelectedAgent(null);
    setIsCreating(false);
  };

  const handleCancel = () => {
    setSelectedAgent(null);
    setIsCreating(false);
  };

  return (
    <Box sx={{ height: '100%' }}>
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* Agent List Panel */}
        <Grid item xs={12} md={4} lg={3}>
          <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Agents</Typography>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={handleCreateAgent}
              >
                Create Agent
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <List>
              {agents.length === 0 ? (
                <ListItem>
                  <ListItemText 
                    primary="No agents defined" 
                    secondary="Click 'Create Agent' to add a new agent"
                  />
                </ListItem>
              ) : (
                agents.map((agent) => (
                  <ListItem
                    key={agent.id}
                    button
                    selected={selectedAgent?.id === agent.id}
                    onClick={() => handleEditAgent(agent)}
                  >
                    <ListItemText
                      primary={agent.name}
                      secondary={agent.description || 'No description'}
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => handleEditAgent(agent)}
                        size="small"
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton
                        edge="end"
                        onClick={() => handleDeleteAgent(agent.id)}
                        color="error"
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))
              )}
            </List>
          </Paper>
        </Grid>

        {/* Agent Editor Panel */}
        <Grid item xs={12} md={8} lg={9}>
          <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
            {isCreating || selectedAgent ? (
              <AgentForm
                initialAgent={selectedAgent}
                onSave={handleSaveAgent}
                onCancel={handleCancel}
                isCreating={isCreating}
              />
            ) : (
              <Box
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <Typography variant="h6" color="text.secondary">
                  Select an agent to edit or create a new one
                </Typography>
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={handleCreateAgent}
                  sx={{ mt: 2 }}
                >
                  Create Agent
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AgentEditor;