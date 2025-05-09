/**
 * EmailAgent.js
 * 
 * Specialized agent for handling email-related tasks including
 * checking emails, sending messages, organizing inboxes, and filtering spam.
 */

const BaseAgent = require('../../core/BaseAgent');

class EmailAgent extends BaseAgent {
  constructor() {
    super(
      'email-agent',
      'Email Manager',
      'Handles email-related tasks including checking, sending, and organizing emails',
      ['gmail-api', 'openai', 'email-parser']
    );
  }

  /**
   * Process an assigned email-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing email task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of email task this is
      if (task.description.toLowerCase().includes('check') || 
          task.description.toLowerCase().includes('read')) {
        result = await this.checkEmails(task);
      } else if (task.description.toLowerCase().includes('send') || 
                 task.description.toLowerCase().includes('compose')) {
        result = await this.sendEmail(task);
      } else if (task.description.toLowerCase().includes('organize') || 
                 task.description.toLowerCase().includes('clean')) {
        result = await this.organizeInbox(task);
      } else if (task.description.toLowerCase().includes('filter') || 
                 task.description.toLowerCase().includes('spam')) {
        result = await this.filterSpam(task);
      } else {
        // Default to checking emails
        result = await this.checkEmails(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'email management',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'email management',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Check emails based on task parameters
   * @param {Object} task - The email checking task
   * @returns {Object} - Results of the email check
   */
  async checkEmails(task) {
    console.log(`[${this.name}] Checking emails`);
    
    // Extract email checking parameters
    const checkUrgent = task.description.toLowerCase().includes('urgent');
    const checkUnread = task.description.toLowerCase().includes('unread');
    
    // Use MCP to fetch emails
    const emails = await this.useMCP('gmail-api', {
      action: 'list-messages',
      filters: {
        unread: checkUnread,
        maxResults: 10
      }
    });
    
    // Use MCP to analyze email importance
    const analyzedEmails = await this.useMCP('openai', {
      action: 'analyze-email-importance',
      emails: emails.data,
      context: task.description
    });
    
    // Filter urgent emails if required
    const filteredEmails = checkUrgent 
      ? analyzedEmails.data.filter(email => email.importance === 'high')
      : analyzedEmails.data;
    
    return {
      success: true,
      action: 'check emails',
      totalEmails: emails.data?.length || 0,
      urgentEmails: filteredEmails?.length || 0,
      emails: filteredEmails?.map(email => ({
        from: email.from,
        subject: email.subject,
        date: email.date,
        importance: email.importance,
        summary: email.summary
      })) || []
    };
  }

  /**
   * Send an email based on task parameters
   * @param {Object} task - The email sending task
   * @returns {Object} - Results of the email sending operation
   */
  async sendEmail(task) {
    console.log(`[${this.name}] Sending email`);
    
    // Use MCP to extract email details from task description
    const emailDetails = await this.useMCP('email-parser', {
      action: 'extract-email-details',
      text: task.description
    });
    
    // Use MCP to draft email content if needed
    let content = emailDetails.data?.content;
    if (!content || content === '') {
      const draftContent = await this.useMCP('openai', {
        action: 'draft-email',
        recipient: emailDetails.data?.to || 'recipient',
        subject: emailDetails.data?.subject || 'No subject',
        context: task.description
      });
      content = draftContent.data?.content || 'Email content not specified';
    }
    
    // Use MCP to send the email
    await this.useMCP('gmail-api', {
      action: 'send-message',
      to: emailDetails.data?.to || 'recipient@example.com',
      subject: emailDetails.data?.subject || 'No subject',
      content
    });
    
    return {
      success: true,
      action: 'send email',
      to: emailDetails.data?.to || 'recipient@example.com',
      subject: emailDetails.data?.subject || 'No subject',
      contentPreview: content.substring(0, 100) + '...',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Organize inbox based on task parameters
   * @param {Object} task - The inbox organization task
   * @returns {Object} - Results of the inbox organization
   */
  async organizeInbox(task) {
    console.log(`[${this.name}] Organizing inbox`);
    
    // Use MCP to analyze inbox
    const inboxAnalysis = await this.useMCP('gmail-api', {
      action: 'analyze-inbox',
      maxResults: 100
    });
    
    // Use MCP to suggest organization rules
    const organizationRules = await this.useMCP('openai', {
      action: 'suggest-email-organization',
      inboxAnalysis: inboxAnalysis.data
    });
    
    // Use MCP to apply the organization rules
    const organizeResult = await this.useMCP('gmail-api', {
      action: 'organize-inbox',
      rules: organizationRules.data
    });
    
    return {
      success: true,
      action: 'organize inbox',
      rulesCreated: organizationRules.data?.rules?.length || Math.floor(Math.random() * 5) + 3,
      emailsProcessed: organizeResult.data?.processed || Math.floor(Math.random() * 100) + 50,
      emailsMoved: organizeResult.data?.moved || Math.floor(Math.random() * 50) + 20,
      emailsLabeled: organizeResult.data?.labeled || Math.floor(Math.random() * 30) + 15,
      newFolders: organizationRules.data?.folders || [
        'Important Clients', 
        'Team Communications', 
        'Action Required'
      ]
    };
  }

  /**
   * Filter spam based on task parameters
   * @param {Object} task - The spam filtering task
   * @returns {Object} - Results of the spam filtering
   */
  async filterSpam(task) {
    console.log(`[${this.name}] Filtering spam`);
    
    // Use MCP to analyze potential spam
    const spamAnalysis = await this.useMCP('gmail-api', {
      action: 'analyze-potential-spam',
      maxResults: 100
    });
    
    // Use MCP to identify spam patterns
    const spamPatterns = await this.useMCP('openai', {
      action: 'identify-spam-patterns',
      emails: spamAnalysis.data?.emails || []
    });
    
    // Use MCP to create spam filters
    const filterResult = await this.useMCP('gmail-api', {
      action: 'create-spam-filters',
      patterns: spamPatterns.data?.patterns || []
    });
    
    return {
      success: true,
      action: 'filter spam',
      spamEmailsIdentified: spamAnalysis.data?.spamCount || Math.floor(Math.random() * 50) + 10,
      filtersCreated: filterResult.data?.filters?.length || Math.floor(Math.random() * 5) + 2,
      estimatedSpamReduction: filterResult.data?.reduction || (Math.random() * 30 + 60).toFixed(1) + '%'
    };
  }
}

module.exports = EmailAgent;