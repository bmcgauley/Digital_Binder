/**
 * BaseAgent.js
 * 
 * Base class that all specialized agents will inherit from.
 * Provides common functionality for task handling and MCP integration.
 */

class BaseAgent {
  constructor(id, name, description, mcps = []) {
    this.id = id;
    this.name = name;
    this.description = description;
    this.mcps = mcps;
    this.activeTasks = [];
    this.coordinator = null;
  }

  /**
   * Connect this agent to the system coordinator
   * @param {Object} coordinator - The system coordinator instance
   */
  connectToCoordinator(coordinator) {
    this.coordinator = coordinator;
    coordinator.registerAgent(this);
  }

  /**
   * Assign a task to this agent
   * @param {Object} task - The task to be assigned
   */
  assignTask(task) {
    console.log(`[${this.name}] Assigned task: ${task.description} (ID: ${task.id})`);
    this.activeTasks.push(task);
    
    // Begin processing the task
    this.processTask(task);
  }

  /**
   * Process an assigned task (to be implemented by subclasses)
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    throw new Error(`processTask() must be implemented by ${this.name}`);
  }

  /**
   * Use an MCP tool to accomplish a task
   * @param {String} mcpName - Name of the MCP to use
   * @param {Object} params - Parameters to pass to the MCP
   * @returns {Promise<Object>} - Results from the MCP
   */
  async useMCP(mcpName, params) {
    console.log(`[${this.name}] Using MCP: ${mcpName}`);
    
    // In a real implementation, this would connect to the actual MCP
    // For this demo, we're just simulating the MCP call
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(`[${this.name}] Completed MCP call: ${mcpName}`);
        resolve({
          success: true,
          data: `Simulated result from ${mcpName}`,
          timestamp: new Date()
        });
      }, 1000);
    });
  }

  /**
   * Report task completion back to the coordinator
   * @param {String} taskId - ID of the completed task
   * @param {Object} result - Results from this agent's work
   */
  completeTask(taskId, result) {
    // Remove from active tasks
    this.activeTasks = this.activeTasks.filter(t => t.id !== taskId);
    
    // Report back to coordinator
    if (this.coordinator) {
      this.coordinator.reportTaskProgress(taskId, this.id, result);
      console.log(`[${this.name}] Reported completion of task ${taskId}`);
    } else {
      console.warn(`[${this.name}] No coordinator to report task completion to`);
    }
  }

  /**
   * Get status information about this agent
   * @returns {Object} - Status information
   */
  getStatus() {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      activeTasks: this.activeTasks.length,
      mcps: this.mcps
    };
  }
}

module.exports = BaseAgent;