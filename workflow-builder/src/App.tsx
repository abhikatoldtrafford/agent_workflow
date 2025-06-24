import React, { useState } from 'react';
import { Box, AppBar, Toolbar, Typography, Tabs, Tab, Button, IconButton, Snackbar, Alert } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import GetAppIcon from '@mui/icons-material/GetApp';

import WorkflowEditor from './components/WorkflowEditor';
import AgentEditor from './components/AgentEditor';
import { WorkflowProvider, useWorkflow } from './components/WorkflowContext';
import { workflowToYAML, agentToYAML, parseWorkflowYAML, parseAgentYAML } from './utils/yaml-converter';
import { saveToFile, readFileAsText, openFileDialog } from './utils/file-utils';

const AppContent: React.FC = () => {
  const [tabIndex, setTabIndex] = useState(0);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success'
  });

  // Access workflow context
  const {
    workflow,
    stages,
    tasks,
    agents,
    updateWorkflow,
    addStage,
    addTask,
    addAgent
  } = useWorkflow();

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const handleExecuteWorkflow = () => {
    // Logic to execute the workflow
    setSnackbar({
      open: true,
      message: 'Workflow execution started',
      severity: 'success'
    });
  };

  const handleSaveWorkflow = () => {
    // In a real app, this would save to a server or database
    setSnackbar({
      open: true,
      message: 'Workflow saved successfully',
      severity: 'success'
    });
  };

  const handleExportYAML = () => {
    try {
      // Generate YAML for the current workflow
      const yamlContent = workflowToYAML(workflow, stages, tasks);
      
      // Save to file and trigger download
      saveToFile(yamlContent, `${workflow.name.toLowerCase().replace(/\s+/g, '-')}.yaml`, 'text/yaml');
      
      setSnackbar({
        open: true,
        message: 'Workflow exported to YAML successfully',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error exporting workflow:', error);
      
      setSnackbar({
        open: true,
        message: `Failed to export workflow: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  };

  const handleExportAgent = () => {
    try {
      if (agents.length === 0) {
        setSnackbar({
          open: true,
          message: 'No agents to export',
          severity: 'error'
        });
        return;
      }

      // For simplicity, just export the first agent
      // In a real app, you'd show a dialog to select which agent to export
      const agent = agents[0];
      const yamlContent = agentToYAML(agent);
      
      // Save to file and trigger download
      saveToFile(yamlContent, `${agent.name.toLowerCase().replace(/\s+/g, '-')}.yaml`, 'text/yaml');
      
      setSnackbar({
        open: true,
        message: `Agent "${agent.name}" exported to YAML successfully`,
        severity: 'success'
      });
    } catch (error) {
      console.error('Error exporting agent:', error);
      
      setSnackbar({
        open: true,
        message: `Failed to export agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  };

  const handleImportYAML = async () => {
    try {
      // Open file dialog and get selected files, allowing multiple files
      const files = await openFileDialog('.yaml,.yml', true);
      
      // Prepare to sort files: agents first, workflows second
      const agentFiles: File[] = [];
      const workflowFiles: File[] = [];
      
      // Separate files into agents and workflows
      for (const file of files) {
        // Read file contents to determine type
        const content = await readFileAsText(file);
        
        // Determine if it's a workflow or agent based on content
        const isWorkflow = content.includes('stages:');
        
        if (isWorkflow) {
          workflowFiles.push(file);
        } else {
          agentFiles.push(file);
        }
      }
      
      // Process agents first (avoiding redundant file reads)
      let agentsImported = 0;
      for (const file of agentFiles) {
        try {
          const content = await readFileAsText(file);
          const agent = parseAgentYAML(content);
          addAgent(agent);
          agentsImported++;
        } catch (err) {
          console.error(`Error importing agent file ${file.name}:`, err);
        }
      }
      
      // Then process workflows
      let workflowsImported = 0;
      for (const file of workflowFiles) {
        try {
          const content = await readFileAsText(file);
          const { workflow: parsedWorkflow, stages: parsedStages, tasks: parsedTasks } = parseWorkflowYAML(content);
          
          // Update the workflow in context if it's the first workflow
          if (workflowsImported === 0) {
            updateWorkflow(parsedWorkflow);
          }
          
          // Add stages and tasks
          parsedStages.forEach(stage => {
            addStage(stage);
          });
          
          parsedTasks.forEach(task => {
            addTask(task);
          });
          
          workflowsImported++;
        } catch (err) {
          console.error(`Error importing workflow file ${file.name}:`, err);
        }
      }
      
      // Determine which tab to switch to and show success message
      if (workflowsImported > 0 && agentsImported > 0) {
        setSnackbar({
          open: true,
          message: `Imported ${workflowsImported} workflow(s) and ${agentsImported} agent(s) successfully`,
          severity: 'success'
        });
        // Keep the current tab
      } else if (workflowsImported > 0) {
        setSnackbar({
          open: true,
          message: `Imported ${workflowsImported} workflow(s) successfully`,
          severity: 'success'
        });
        setTabIndex(0);
      } else if (agentsImported > 0) {
        setSnackbar({
          open: true,
          message: `Imported ${agentsImported} agent(s) successfully`,
          severity: 'success'
        });
        setTabIndex(1);
      }
    } catch (error) {
      console.error('Error importing YAML:', error);
      
      setSnackbar({
        open: true,
        message: `Failed to import YAML: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Workflow Builder
          </Typography>
          <Button 
            variant="contained" 
            color="success" 
            startIcon={<PlayArrowIcon />}
            onClick={handleExecuteWorkflow}
            sx={{ mr: 1 }}
          >
            Execute
          </Button>
          <IconButton color="inherit" onClick={handleSaveWorkflow} title="Save Workflow">
            <SaveIcon />
          </IconButton>
          
          <Box sx={{ position: 'relative', display: 'inline-block' }}>
            <IconButton 
              color="inherit" 
              onClick={() => tabIndex === 0 ? handleExportYAML() : handleExportAgent()}
              title={tabIndex === 0 ? "Export Workflow as YAML" : "Export Agent as YAML"}
            >
              <GetAppIcon />
            </IconButton>
          </Box>
          
          <IconButton color="inherit" onClick={handleImportYAML} title="Import YAML">
            <FileUploadIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      <Box sx={{ bgcolor: 'background.paper', mt: 1 }}>
        <Tabs value={tabIndex} onChange={handleTabChange} centered>
          <Tab label="Workflow Editor" />
          <Tab label="Agent Editor" />
        </Tabs>
      </Box>

      <Box sx={{ p: 2, height: 'calc(100vh - 120px)' }}>
        {tabIndex === 0 ? (
          <WorkflowEditor />
        ) : (
          <AgentEditor />
        )}
      </Box>

      {/* Notification system */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

const App: React.FC = () => {
  return (
    <WorkflowProvider>
      <AppContent />
    </WorkflowProvider>
  );
}

export default App;