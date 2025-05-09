/**
 * SystemsAgent.js
 * 
 * Specialized agent for handling systems-related tasks including
 * systems analysis, systems design, requirements gathering,
 * and technical architecture.
 */

const BaseAgent = require('../../core/BaseAgent');

class SystemsAgent extends BaseAgent {
  constructor() {
    super(
      'systems-agent',
      'Systems Specialist',
      'Handles systems analysis, design, architecture, and requirements engineering',
      ['openai', 'system-modeling', 'requirements-analyzer', 'architecture-planner']
    );
  }

  /**
   * Process an assigned systems-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing systems task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of systems task this is
      if (task.description.toLowerCase().includes('analyze system') || 
          task.description.toLowerCase().includes('system analysis')) {
        result = await this.analyzeSystem(task);
      } else if (task.description.toLowerCase().includes('design system') || 
                 task.description.toLowerCase().includes('system design')) {
        result = await this.designSystem(task);
      } else if (task.description.toLowerCase().includes('requirements') || 
                 task.description.toLowerCase().includes('gather requirements')) {
        result = await this.gatherRequirements(task);
      } else if (task.description.toLowerCase().includes('architecture') || 
                 task.description.toLowerCase().includes('technical architecture')) {
        result = await this.developArchitecture(task);
      } else {
        // Default to general systems analysis
        result = await this.analyzeSystem(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'systems engineering',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'systems engineering',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Analyze a system based on task parameters
   * @param {Object} task - The system analysis task
   * @returns {Object} - Results of the system analysis
   */
  async analyzeSystem(task) {
    console.log(`[${this.name}] Analyzing system`);
    
    // Extract system information from task
    const systemName = this.extractSystemName(task.description);
    
    // Use MCP to gather system information
    const systemInfo = await this.useMCP('system-modeling', {
      action: 'analyze-current-state',
      systemName,
      context: task.description
    });
    
    // Use MCP to identify pain points
    const painPoints = await this.useMCP('requirements-analyzer', {
      action: 'identify-issues',
      systemData: systemInfo.data
    });
    
    // Use MCP to analyze system strengths and weaknesses
    const analysis = await this.useMCP('openai', {
      action: 'system-analysis',
      systemInfo: systemInfo.data,
      painPoints: painPoints.data,
      context: task.description
    });
    
    return {
      success: true,
      action: 'analyze system',
      systemName,
      currentState: {
        modules: ['User Management', 'Data Processing', 'Reporting', 'Administration'],
        technologies: ['Java', 'Oracle', 'REST APIs', 'Angular'],
        maturity: 'Established but aging'
      },
      painPoints: painPoints.data?.issues || [
        'Performance bottlenecks in data processing module',
        'Scalability issues during peak times',
        'Poor integration with modern systems',
        'High maintenance costs'
      ],
      recommendations: analysis.data?.recommendations || [
        'Modernize data processing pipeline',
        'Implement caching strategy',
        'Adopt microservices architecture for critical components',
        'Enhance monitoring and observability'
      ]
    };
  }

  /**
   * Design a system based on task parameters
   * @param {Object} task - The system design task
   * @returns {Object} - Results of the system design
   */
  async designSystem(task) {
    console.log(`[${this.name}] Designing system`);
    
    // Extract system requirements from task
    const systemName = this.extractSystemName(task.description);
    
    // Use MCP to analyze requirements
    const requirements = await this.useMCP('requirements-analyzer', {
      action: 'extract-requirements',
      text: task.description,
      context: 'system design'
    });
    
    // Use MCP to generate system design
    const design = await this.useMCP('system-modeling', {
      action: 'design-system',
      systemName,
      requirements: requirements.data
    });
    
    // Use MCP to validate the design
    const validation = await this.useMCP('architecture-planner', {
      action: 'validate-design',
      design: design.data,
      requirements: requirements.data
    });
    
    return {
      success: true,
      action: 'design system',
      systemName,
      architecture: design.data?.architecture || 'Microservices',
      components: design.data?.components || [
        {
          name: 'User Service',
          responsibility: 'Authentication and authorization',
          technology: 'Node.js, JWT'
        },
        {
          name: 'Data Processing Service',
          responsibility: 'Handle core business logic',
          technology: 'Python, Apache Kafka'
        },
        {
          name: 'Analytics Service',
          responsibility: 'Reporting and analytics',
          technology: 'Python, TensorFlow, ElasticSearch'
        },
        {
          name: 'Frontend Application',
          responsibility: 'User interface',
          technology: 'React, Redux'
        }
      ],
      dataFlow: design.data?.dataFlow || 'REST APIs with event-driven architecture for asynchronous processes',
      validationResults: validation.data?.results || {
        passedRequirements: '92%',
        concerns: ['Potential latency in analytics processing', 'Consider backup strategy for data pipelines']
      }
    };
  }

  /**
   * Gather requirements based on task parameters
   * @param {Object} task - The requirements gathering task
   * @returns {Object} - Results of the requirements gathering
   */
  async gatherRequirements(task) {
    console.log(`[${this.name}] Gathering requirements`);
    
    // Extract system information from task
    const systemName = this.extractSystemName(task.description);
    
    // Use MCP to analyze stakeholders
    const stakeholders = await this.useMCP('requirements-analyzer', {
      action: 'identify-stakeholders',
      context: task.description
    });
    
    // Use MCP to extract requirements
    const requirements = await this.useMCP('requirements-analyzer', {
      action: 'extract-requirements',
      text: task.description,
      stakeholders: stakeholders.data
    });
    
    // Use MCP to organize and prioritize requirements
    const organizedRequirements = await this.useMCP('openai', {
      action: 'organize-requirements',
      requirements: requirements.data,
      stakeholders: stakeholders.data
    });
    
    return {
      success: true,
      action: 'gather requirements',
      systemName,
      stakeholders: stakeholders.data?.list || [
        { role: 'End Users', interests: ['Usability', 'Performance', 'Reliability'] },
        { role: 'System Administrators', interests: ['Maintainability', 'Monitoring', 'Security'] },
        { role: 'Business Leaders', interests: ['Cost-effectiveness', 'Scalability', 'Analytics'] }
      ],
      functionalRequirements: organizedRequirements.data?.functional || [
        'User authentication and authorization',
        'Data processing pipeline',
        'Reporting and analytics dashboard',
        'API integration with external systems',
        'Administrative control panel'
      ],
      nonFunctionalRequirements: organizedRequirements.data?.nonFunctional || [
        'System should handle 1000+ concurrent users',
        '99.9% uptime requirement',
        'Data processing completes within 3 seconds',
        'Compliance with industry security standards',
        'Support for multiple languages'
      ],
      prioritization: organizedRequirements.data?.priorities || {
        'high': ['User authentication', 'Data processing core functions'],
        'medium': ['Reporting capabilities', 'External API integration'],
        'low': ['Administrative features', 'Advanced analytics']
      }
    };
  }

  /**
   * Develop system architecture based on task parameters
   * @param {Object} task - The architecture development task
   * @returns {Object} - Results of the architecture development
   */
  async developArchitecture(task) {
    console.log(`[${this.name}] Developing architecture`);
    
    // Extract system information from task
    const systemName = this.extractSystemName(task.description);
    
    // Use MCP to analyze requirements
    const requirements = await this.useMCP('requirements-analyzer', {
      action: 'extract-requirements',
      text: task.description,
      context: 'system architecture'
    });
    
    // Use MCP to select architectural patterns
    const patterns = await this.useMCP('architecture-planner', {
      action: 'recommend-patterns',
      requirements: requirements.data
    });
    
    // Use MCP to develop full architecture
    const architecture = await this.useMCP('architecture-planner', {
      action: 'design-architecture',
      systemName,
      requirements: requirements.data,
      patterns: patterns.data
    });
    
    return {
      success: true,
      action: 'develop architecture',
      systemName,
      architecturalStyle: architecture.data?.style || 'Microservices with Event-Driven Integration',
      layers: architecture.data?.layers || [
        'Presentation Layer (React frontend)',
        'API Gateway Layer (Node.js)',
        'Service Layer (Domain-specific microservices)',
        'Data Layer (Polyglot persistence)'
      ],
      components: architecture.data?.components || [
        'Authentication Service',
        'User Profile Service',
        'Transaction Processing Service',
        'Notification Service',
        'Analytics Engine',
        'Data Storage Services'
      ],
      patterns: patterns.data?.recommendations || [
        'Circuit Breaker for resilience',
        'CQRS for scalable data operations',
        'API Gateway for frontend communication',
        'Event Sourcing for audit trail'
      ],
      technologies: architecture.data?.technologies || {
        frontend: ['React', 'Redux', 'Material UI'],
        backend: ['Node.js', 'Java Spring Boot', 'Python Flask'],
        data: ['PostgreSQL', 'MongoDB', 'Redis', 'Kafka'],
        devops: ['Docker', 'Kubernetes', 'Prometheus', 'ELK Stack']
      }
    };
  }

  /**
   * Extract system name from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted system name
   */
  extractSystemName(description) {
    // In a real implementation, this would use NLP to extract the system name
    // For now, look for keywords and get the next few words
    
    const systemKeywords = ['system', 'platform', 'application', 'software', 'solution'];
    const words = description.split(' ');
    
    for (let i = 0; i < words.length - 1; i++) {
      if (systemKeywords.includes(words[i].toLowerCase())) {
        // Get the next 1-3 words as the system name
        return [words[i+1], words[i+2], words[i+3]]
          .filter(w => w && !['for', 'the', 'a', 'an'].includes(w.toLowerCase()))
          .slice(0, 2)
          .join(' ');
      }
    }
    
    return 'Enterprise System'; // Default if no system name found
  }
}

module.exports = SystemsAgent;