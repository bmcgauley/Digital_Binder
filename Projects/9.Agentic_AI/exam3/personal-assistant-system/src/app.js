/**
 * app.js
 * 
 * Main application file that initializes the personal assistant system
 * with all its specialized agents and provides an interface for submitting tasks.
 */

const SystemCoordinator = require('./core/SystemCoordinator');
const EmailAgent = require('./agents/EmailAgent');
const CalendarAgent = require('./agents/CalendarAgent');
const DocumentAgent = require('./agents/DocumentAgent');
const BusinessAgent = require('./agents/BusinessAgent');
const DevelopmentAgent = require('./agents/DevelopmentAgent');
const SystemsAgent = require('./agents/SystemsAgent');
const ResearchAgent = require('./agents/ResearchAgent');

// Create a simple express server to provide an API interface (in a real implementation)
const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

// Enable JSON body parsing
app.use(express.json());

// Initialize the system coordinator
const coordinator = new SystemCoordinator();

// Initialize all specialized agents
const emailAgent = new EmailAgent();
const calendarAgent = new CalendarAgent();
const documentAgent = new DocumentAgent();
const businessAgent = new BusinessAgent();
const developmentAgent = new DevelopmentAgent();
const systemsAgent = new SystemsAgent();
const researchAgent = new ResearchAgent();

// Connect all agents to the coordinator
emailAgent.connectToCoordinator(coordinator);
calendarAgent.connectToCoordinator(coordinator);
documentAgent.connectToCoordinator(coordinator);
businessAgent.connectToCoordinator(coordinator);
developmentAgent.connectToCoordinator(coordinator);
systemsAgent.connectToCoordinator(coordinator);
researchAgent.connectToCoordinator(coordinator);

console.log('Personal Assistant System initialized with 7 specialized agents:');
console.log('1. Email Manager - Handles email tasks');
console.log('2. Calendar Manager - Handles scheduling and reminders');
console.log('3. Document Specialist - Prepares and reviews papers and documents');
console.log('4. Business Analyst - Analyzes client businesses and markets');
console.log('5. Web Developer - Codes and debugs websites and applications');
console.log('6. Systems Specialist - Performs systems analysis and design');
console.log('7. Research Specialist - Gathers information and performs research');

// API endpoint to submit a task to the personal assistant system
app.post('/task', (req, res) => {
  const { description, requirements, deadline } = req.body;
  
  if (!description) {
    return res.status(400).json({
      success: false,
      error: 'Task description is required'
    });
  }
  
  const taskId = coordinator.submitTask({
    description,
    requirements,
    deadline
  });
  
  res.status(201).json({
    success: true,
    message: 'Task submitted successfully',
    taskId
  });
});

// API endpoint to check task status
app.get('/task/:taskId', (req, res) => {
  const { taskId } = req.params;
  
  const status = coordinator.getTaskStatus(taskId);
  
  if (status.status === 'not-found') {
    return res.status(404).json({
      success: false,
      error: 'Task not found'
    });
  }
  
  res.json({
    success: true,
    status
  });
});

// API endpoint to list all agents
app.get('/agents', (req, res) => {
  const agents = coordinator.agents.map(agent => ({
    id: agent.id,
    name: agent.name,
    description: agent.description,
    mcps: agent.mcps
  }));
  
  res.json({
    success: true,
    agents
  });
});

// Simple demonstration function to simulate task submission directly
function simulateTask(description) {
  console.log(`\n--- Simulating task: "${description}" ---`);
  
  const taskId = coordinator.submitTask({
    description,
    requirements: 'Complete the task efficiently using appropriate MCPs',
    deadline: new Date(Date.now() + 3600000) // 1 hour from now
  });
  
  console.log(`Task submitted with ID: ${taskId}`);
  
  // In a real implementation, you'd set up proper async handling
  // For this demo, we're just letting the system do its work
  
  // Check status after a short delay
  setTimeout(() => {
    const status = coordinator.getTaskStatus(taskId);
    console.log(`\nTask status: ${status.status}`);
    
    if (status.status === 'completed') {
      console.log('Task completed successfully!');
      console.log('Results:', status.result);
    } else {
      console.log(`Progress: ${Math.round(status.progress * 100)}%`);
      console.log('Completed steps:', status.completedSteps.join(', '));
    }
  }, 5000);
}

// Start the server
if (require.main === module) {
  app.listen(port, () => {
    console.log(`Personal Assistant System API running on port ${port}`);
    
    // Simulate a few example tasks if running directly
    simulateTask('Check my email for any urgent messages');
    
    setTimeout(() => {
      simulateTask('Analyze the client business proposal and prepare a report');
    }, 2000);
    
    setTimeout(() => {
      simulateTask('Design a system architecture for a new e-commerce platform');
    }, 4000);
  });
} else {
  // Export for use in other files
  module.exports = {
    coordinator,
    app
  };
}