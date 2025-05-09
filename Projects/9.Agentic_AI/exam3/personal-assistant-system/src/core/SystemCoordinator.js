/**
 * SystemCoordinator.js
 * 
 * This is the main coordinator for the personal assistant system.
 * It orchestrates communication between different specialized agents
 * and handles task distribution and result aggregation.
 */

class SystemCoordinator {
  constructor(agents = []) {
    this.agents = agents;
    this.taskQueue = [];
    this.results = new Map();
    this.activeTaskId = null;
  }

  /**
   * Register an agent with the coordinator
   * @param {Object} agent - The agent to register
   */
  registerAgent(agent) {
    if (!this.agents.some(a => a.id === agent.id)) {
      this.agents.push(agent);
      console.log(`Registered agent: ${agent.name} (${agent.id})`);
    } else {
      console.warn(`Agent with ID ${agent.id} is already registered`);
    }
  }

  /**
   * Submit a new task to the system
   * @param {Object} task - Task details including description, requirements, and deadline
   * @returns {String} - Task ID
   */
  submitTask(task) {
    const taskId = `task-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    this.taskQueue.push({
      id: taskId,
      ...task,
      status: 'pending',
      submittedAt: new Date(),
      assignedAgents: [],
      completedSteps: [],
      results: {}
    });
    
    console.log(`Task submitted: ${task.description} (ID: ${taskId})`);
    
    // Process the queue if we're not already working on something
    if (!this.activeTaskId) {
      this.processNextTask();
    }
    
    return taskId;
  }

  /**
   * Process the next task in the queue
   */
  processNextTask() {
    if (this.taskQueue.length === 0) {
      console.log('No tasks in the queue');
      this.activeTaskId = null;
      return;
    }
    
    const task = this.taskQueue[0];
    this.activeTaskId = task.id;
    task.status = 'in-progress';
    
    console.log(`Processing task: ${task.description} (ID: ${task.id})`);
    
    // Analyze the task to determine which agents should be involved
    const requiredAgents = this.analyzeTaskRequirements(task);
    
    // Assign the task to appropriate agents
    requiredAgents.forEach(agentId => {
      const agent = this.agents.find(a => a.id === agentId);
      if (agent) {
        task.assignedAgents.push(agentId);
        agent.assignTask(task);
      }
    });
    
    // For this implementation, we're assuming synchronous operation
    // In a real system, you'd implement proper async handling and callbacks
  }

  /**
   * Analyze task to determine which agents should be involved
   * @param {Object} task - The task to analyze
   * @returns {Array} - Array of agent IDs that should work on this task
   */
  analyzeTaskRequirements(task) {
    const keywords = {
      'email': 'email-agent',
      'mail': 'email-agent',
      'calendar': 'calendar-agent',
      'schedule': 'calendar-agent',
      'paper': 'document-agent',
      'document': 'document-agent',
      'review': 'document-agent',
      'business': 'business-agent',
      'client': 'business-agent',
      'analyze': 'business-agent',
      'code': 'development-agent',
      'website': 'development-agent',
      'debug': 'development-agent',
      'system': 'systems-agent',
      'analysis': 'systems-agent',
      'design': 'systems-agent'
    };

    // Default to using the research agent for any task
    const requiredAgents = new Set(['research-agent']);
    
    // Look for keywords in the task description and requirements
    const taskText = `${task.description} ${task.requirements || ''}`.toLowerCase();
    
    Object.entries(keywords).forEach(([keyword, agentId]) => {
      if (taskText.includes(keyword)) {
        requiredAgents.add(agentId);
      }
    });
    
    return Array.from(requiredAgents);
  }

  /**
   * Called by agents when they complete their part of a task
   * @param {String} taskId - ID of the completed task
   * @param {String} agentId - ID of the agent that completed its portion
   * @param {Object} result - Results from the agent
   */
  reportTaskProgress(taskId, agentId, result) {
    const taskIndex = this.taskQueue.findIndex(t => t.id === taskId);
    
    if (taskIndex === -1) {
      console.warn(`Task ${taskId} not found in queue`);
      return;
    }
    
    const task = this.taskQueue[taskIndex];
    
    // Store the result
    task.results[agentId] = result;
    task.completedSteps.push(agentId);
    
    console.log(`Agent ${agentId} completed their part of task ${taskId}`);
    
    // Check if all assigned agents have completed their work
    const allAgentsComplete = task.assignedAgents.every(id => 
      task.completedSteps.includes(id));
    
    if (allAgentsComplete) {
      this.finalizeTask(taskIndex);
    }
  }

  /**
   * Finalize a task when all agents have completed their parts
   * @param {Number} taskIndex - Index of the task in the queue
   */
  finalizeTask(taskIndex) {
    const task = this.taskQueue[taskIndex];
    task.status = 'completed';
    task.completedAt = new Date();
    
    console.log(`Task completed: ${task.description} (ID: ${task.id})`);
    
    // Store the full results
    this.results.set(task.id, {
      task,
      results: task.results
    });
    
    // Remove the task from the queue
    this.taskQueue.splice(taskIndex, 1);
    
    // Process the next task if there is one
    this.activeTaskId = null;
    this.processNextTask();
    
    return task;
  }

  /**
   * Get the status of a specific task
   * @param {String} taskId - ID of the task
   * @returns {Object} - Task status information
   */
  getTaskStatus(taskId) {
    const activeTask = this.taskQueue.find(t => t.id === taskId);
    
    if (activeTask) {
      return {
        status: activeTask.status,
        progress: activeTask.completedSteps.length / activeTask.assignedAgents.length,
        completedSteps: activeTask.completedSteps
      };
    }
    
    // Check if it's a completed task
    if (this.results.has(taskId)) {
      return {
        status: 'completed',
        progress: 1,
        result: this.results.get(taskId)
      };
    }
    
    return { status: 'not-found' };
  }
}

module.exports = SystemCoordinator;