/**
 * DocumentAgent.js
 * 
 * Specialized agent for handling document-related tasks including
 * preparing papers, reviewing documents, generating reports, and formatting.
 */

const BaseAgent = require('../../core/BaseAgent');

class DocumentAgent extends BaseAgent {
  constructor() {
    super(
      'document-agent',
      'Document Specialist',
      'Handles document-related tasks including paper preparation, reviewing, and formatting',
      ['openai', 'scholar-search', 'document-formatter', 'grammar-checker']
    );
    
    this.supportedDocumentTypes = ['academic', 'business', 'technical', 'creative'];
  }

  /**
   * Process an assigned document-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing document task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of document task this is
      if (task.description.toLowerCase().includes('prepare') || 
          task.description.toLowerCase().includes('write')) {
        result = await this.prepareDocument(task);
      } else if (task.description.toLowerCase().includes('review') || 
                 task.description.toLowerCase().includes('analyze')) {
        result = await this.reviewDocument(task);
      } else if (task.description.toLowerCase().includes('format') || 
                 task.description.toLowerCase().includes('style')) {
        result = await this.formatDocument(task);
      } else if (task.description.toLowerCase().includes('research') || 
                 task.description.toLowerCase().includes('find sources')) {
        result = await this.researchTopic(task);
      } else {
        // Default to a general document preparation
        result = await this.prepareDocument(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'document processing',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'document processing',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Prepare a document based on task parameters
   * @param {Object} task - The document preparation task
   * @returns {Object} - Results of the document preparation
   */
  async prepareDocument(task) {
    console.log(`[${this.name}] Preparing document`);
    
    // Determine the document type
    let documentType = 'general';
    const taskText = task.description.toLowerCase();
    
    if (taskText.includes('academic') || taskText.includes('paper') || taskText.includes('thesis')) {
      documentType = 'academic';
    } else if (taskText.includes('business') || taskText.includes('proposal') || taskText.includes('report')) {
      documentType = 'business';
    } else if (taskText.includes('technical') || taskText.includes('manual')) {
      documentType = 'technical';
    } else if (taskText.includes('creative') || taskText.includes('story')) {
      documentType = 'creative';
    }
    
    // Research the topic using MCP
    const research = await this.useMCP('scholar-search', {
      action: 'search',
      query: task.description,
      documentType,
      limit: 5
    });
    
    // Generate content using MCP
    const content = await this.useMCP('openai', {
      action: 'generate-document',
      topic: task.description,
      type: documentType,
      research: research.data,
      length: 'medium'
    });
    
    // Format the document using MCP
    const formattedDocument = await this.useMCP('document-formatter', {
      action: 'format',
      content: content.data,
      style: this.getStyleForDocumentType(documentType)
    });
    
    return {
      success: true,
      action: 'prepare document',
      documentType,
      title: this.extractTitle(task.description),
      wordCount: Math.floor(Math.random() * 2000) + 1000,
      sections: ['Introduction', 'Body', 'Conclusion'],
      sources: Math.floor(Math.random() * 10) + 5
    };
  }

  /**
   * Review a document based on task parameters
   * @param {Object} task - The document review task
   * @returns {Object} - Results of the document review
   */
  async reviewDocument(task) {
    console.log(`[${this.name}] Reviewing document`);
    
    // Extract document content from task
    // In a real implementation, this would be a file path or content
    const documentContent = task.documentContent || 'Sample document content for review';
    
    // Use MCP to check grammar and style
    const grammarCheck = await this.useMCP('grammar-checker', {
      action: 'check',
      text: documentContent
    });
    
    // Use MCP to check content quality
    const contentReview = await this.useMCP('openai', {
      action: 'review-document',
      content: documentContent,
      criteria: ['clarity', 'coherence', 'evidence', 'argument']
    });
    
    return {
      success: true,
      action: 'review document',
      grammarIssues: Math.floor(Math.random() * 10),
      styleIssues: Math.floor(Math.random() * 5),
      contentScore: Math.floor(Math.random() * 30) + 70, // 70-100 score
      suggestions: [
        'Improve clarity in section 2',
        'Add more supporting evidence for main argument',
        'Restructure introduction for better flow'
      ]
    };
  }

  /**
   * Format a document based on task parameters
   * @param {Object} task - The document formatting task
   * @returns {Object} - Results of the document formatting
   */
  async formatDocument(task) {
    console.log(`[${this.name}] Formatting document`);
    
    // Extract document content and style from task
    const documentContent = task.documentContent || 'Sample document content for formatting';
    let style = 'default';
    
    if (task.description.toLowerCase().includes('apa')) {
      style = 'apa';
    } else if (task.description.toLowerCase().includes('mla')) {
      style = 'mla';
    } else if (task.description.toLowerCase().includes('chicago')) {
      style = 'chicago';
    } else if (task.description.toLowerCase().includes('ieee')) {
      style = 'ieee';
    }
    
    // Use MCP to format the document
    await this.useMCP('document-formatter', {
      action: 'format',
      content: documentContent,
      style
    });
    
    return {
      success: true,
      action: 'format document',
      appliedStyle: style,
      formattedSections: ['Title', 'Abstract', 'Body', 'References'],
      citations: Math.floor(Math.random() * 15) + 5
    };
  }

  /**
   * Research a topic based on task parameters
   * @param {Object} task - The research task
   * @returns {Object} - Results of the research
   */
  async researchTopic(task) {
    console.log(`[${this.name}] Researching topic`);
    
    // Extract topic from task description
    const topic = task.description.replace(/research|find sources|information about|look up/gi, '').trim();
    
    // Use MCP to search for academic sources
    const academicSources = await this.useMCP('scholar-search', {
      action: 'search',
      query: topic,
      limit: 5
    });
    
    // Use MCP to search for general information
    const generalInfo = await this.useMCP('openai', {
      action: 'research',
      topic,
      depth: 'comprehensive'
    });
    
    return {
      success: true,
      action: 'research topic',
      topic,
      academicSourcesFound: Math.floor(Math.random() * 10) + 5,
      keyFindings: [
        'Finding 1: Important insight about the topic',
        'Finding 2: Statistical data supporting the main thesis',
        'Finding 3: Contrasting viewpoints in current literature'
      ]
    };
  }

  /**
   * Helper method to get formatting style based on document type
   * @param {String} documentType - Type of document
   * @returns {String} - Formatting style
   */
  getStyleForDocumentType(documentType) {
    const styleMap = {
      'academic': 'apa',
      'business': 'business',
      'technical': 'ieee',
      'creative': 'standard'
    };
    
    return styleMap[documentType] || 'standard';
  }

  /**
   * Extract a title from the task description
   * @param {String} description - Task description
   * @returns {String} - Extracted title
   */
  extractTitle(description) {
    // In a real implementation, this would use NLP to extract a meaningful title
    // For now, just take the first few words of the description
    const words = description.split(' ');
    return words.slice(0, 4).join(' ') + '...';
  }
}

module.exports = DocumentAgent;