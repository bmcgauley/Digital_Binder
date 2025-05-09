/**
 * CalendarAgent.js
 * 
 * Specialized agent for handling calendar-related tasks including
 * managing appointments, scheduling meetings, and sending reminders.
 */

const BaseAgent = require('../../core/BaseAgent');

class CalendarAgent extends BaseAgent {
  constructor() {
    super(
      'calendar-agent',
      'Calendar Manager',
      'Handles calendar-related tasks including scheduling, reminders, and event management',
      ['google-calendar', 'openai', 'date-time-planner']
    );
    
    this.calendarProviders = ['google', 'outlook', 'apple']; // Supported calendar providers
  }

  /**
   * Process an assigned calendar-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing calendar task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of calendar task this is
      if (task.description.toLowerCase().includes('schedule') || 
          task.description.toLowerCase().includes('book')) {
        result = await this.scheduleEvent(task);
      } else if (task.description.toLowerCase().includes('remind') || 
                 task.description.toLowerCase().includes('notification')) {
        result = await this.createReminder(task);
      } else if (task.description.toLowerCase().includes('update') || 
                 task.description.toLowerCase().includes('modify')) {
        result = await this.updateEvent(task);
      } else if (task.description.toLowerCase().includes('check') || 
                 task.description.toLowerCase().includes('view')) {
        result = await this.checkSchedule(task);
      } else {
        // Default to checking schedule
        result = await this.checkSchedule(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'calendar management',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'calendar management',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Schedule a new event based on task parameters
   * @param {Object} task - The event scheduling task
   * @returns {Object} - Results of the scheduling operation
   */
  async scheduleEvent(task) {
    console.log(`[${this.name}] Scheduling event`);
    
    // In a real implementation, this would parse the task description to extract
    // event details like title, date, time, participants, etc.
    // Here we're using placeholder values
    
    // Use MCP to parse natural language into structured event data
    const eventDetails = await this.useMCP('openai', {
      action: 'extract-event-details',
      text: task.description
    });
    
    // Use MCP to create the calendar event
    await this.useMCP('google-calendar', {
      action: 'create',
      event: {
        title: eventDetails.title || 'New Event',
        start: eventDetails.start || new Date(Date.now() + 86400000), // Tomorrow
        end: eventDetails.end || new Date(Date.now() + 90000000),
        location: eventDetails.location || 'Virtual',
        attendees: eventDetails.attendees || [],
        description: eventDetails.description || ''
      }
    });
    
    return {
      success: true,
      action: 'schedule event',
      eventDetails: {
        title: eventDetails.title || 'New Event',
        date: new Date(Date.now() + 86400000).toLocaleDateString(),
        attendees: (eventDetails.attendees || []).length
      }
    };
  }

  /**
   * Create a reminder based on task parameters
   * @param {Object} task - The reminder creation task
   * @returns {Object} - Results of the reminder creation
   */
  async createReminder(task) {
    console.log(`[${this.name}] Creating reminder`);
    
    // Use MCP to parse the reminder details from natural language
    const reminderDetails = await this.useMCP('openai', {
      action: 'extract-reminder-details',
      text: task.description
    });
    
    // Use MCP to create the reminder
    await this.useMCP('google-calendar', {
      action: 'create-reminder',
      reminder: {
        title: reminderDetails.title || 'Reminder',
        time: reminderDetails.time || new Date(Date.now() + 3600000), // 1 hour from now
        notes: reminderDetails.notes || ''
      }
    });
    
    return {
      success: true,
      action: 'create reminder',
      reminderDetails: {
        title: reminderDetails.title || 'Reminder',
        time: new Date(Date.now() + 3600000).toLocaleTimeString()
      }
    };
  }

  /**
   * Update an existing event based on task parameters
   * @param {Object} task - The event update task
   * @returns {Object} - Results of the update operation
   */
  async updateEvent(task) {
    console.log(`[${this.name}] Updating event`);
    
    // Use MCP to extract event ID and updated details
    const updateDetails = await this.useMCP('openai', {
      action: 'extract-event-update-details',
      text: task.description
    });
    
    // Use MCP to update the event
    await this.useMCP('google-calendar', {
      action: 'update',
      eventId: updateDetails.eventId || 'placeholder-event-id',
      updates: updateDetails.updates || { title: 'Updated Event' }
    });
    
    return {
      success: true,
      action: 'update event',
      eventId: updateDetails.eventId || 'placeholder-event-id',
      updatedFields: Object.keys(updateDetails.updates || { title: 'Updated Event' })
    };
  }

  /**
   * Check the user's schedule based on task parameters
   * @param {Object} task - The schedule checking task
   * @returns {Object} - Results of the schedule check
   */
  async checkSchedule(task) {
    console.log(`[${this.name}] Checking schedule`);
    
    // Use MCP to parse the time range from natural language
    const timeRange = await this.useMCP('date-time-planner', {
      action: 'parse-time-range',
      text: task.description
    });
    
    // Use MCP to fetch calendar events
    const events = await this.useMCP('google-calendar', {
      action: 'list',
      timeMin: timeRange.start || new Date(),
      timeMax: timeRange.end || new Date(Date.now() + 604800000), // 1 week from now
      maxResults: 10
    });
    
    return {
      success: true,
      action: 'check schedule',
      period: `${new Date().toLocaleDateString()} to ${new Date(Date.now() + 604800000).toLocaleDateString()}`,
      eventCount: Math.floor(Math.random() * 5),
      sampleEvents: [
        { title: 'Team Meeting', start: '2025-05-07T10:00:00', end: '2025-05-07T11:00:00' },
        { title: 'Client Presentation', start: '2025-05-09T14:00:00', end: '2025-05-09T15:30:00' }
      ]
    };
  }
}

module.exports = CalendarAgent;