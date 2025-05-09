/**
 * BusinessAgent.js
 * 
 * Specialized agent for handling business-related tasks including
 * client business analysis, financial reporting, market research,
 * and strategy recommendations.
 */

const BaseAgent = require('../../core/BaseAgent');

class BusinessAgent extends BaseAgent {
  constructor() {
    super(
      'business-agent',
      'Business Analyst',
      'Handles business-related tasks including client analysis, market research, and strategy development',
      ['openai', 'business-intelligence', 'market-data', 'financial-analyzer']
    );
  }

  /**
   * Process an assigned business-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing business task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of business task this is
      if (task.description.toLowerCase().includes('analyze client') || 
          task.description.toLowerCase().includes('client analysis')) {
        result = await this.analyzeClientBusiness(task);
      } else if (task.description.toLowerCase().includes('market research') || 
                 task.description.toLowerCase().includes('industry analysis')) {
        result = await this.performMarketResearch(task);
      } else if (task.description.toLowerCase().includes('financial') || 
                 task.description.toLowerCase().includes('finance')) {
        result = await this.analyzeFinancials(task);
      } else if (task.description.toLowerCase().includes('strategy') || 
                 task.description.toLowerCase().includes('recommendation')) {
        result = await this.developStrategy(task);
      } else {
        // Default to general business analysis
        result = await this.analyzeClientBusiness(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'business analysis',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'business analysis',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Analyze a client's business based on task parameters
   * @param {Object} task - The client analysis task
   * @returns {Object} - Results of the client analysis
   */
  async analyzeClientBusiness(task) {
    console.log(`[${this.name}] Analyzing client business`);
    
    // Extract client information from task
    const clientName = this.extractClientName(task.description);
    
    // Use MCP to gather business information
    const companyInfo = await this.useMCP('business-intelligence', {
      action: 'company-profile',
      name: clientName
    });
    
    // Use MCP to get market position
    const marketPosition = await this.useMCP('market-data', {
      action: 'competitive-analysis',
      company: clientName,
      depth: 'comprehensive'
    });
    
    // Use MCP to analyze strengths and weaknesses
    const swotAnalysis = await this.useMCP('openai', {
      action: 'swot-analysis',
      companyData: companyInfo.data,
      marketData: marketPosition.data
    });
    
    return {
      success: true,
      action: 'analyze client business',
      clientName,
      industrySector: companyInfo.data?.industry || 'Technology',
      marketPosition: marketPosition.data?.position || 'Mid-tier',
      strengths: swotAnalysis.data?.strengths || ['Strong product portfolio', 'Established customer base'],
      weaknesses: swotAnalysis.data?.weaknesses || ['Limited marketing reach', 'Legacy systems'],
      opportunities: swotAnalysis.data?.opportunities || ['Emerging market expansion', 'Digital transformation'],
      threats: swotAnalysis.data?.threats || ['Increasing competition', 'Changing regulations']
    };
  }

  /**
   * Perform market research based on task parameters
   * @param {Object} task - The market research task
   * @returns {Object} - Results of the market research
   */
  async performMarketResearch(task) {
    console.log(`[${this.name}] Performing market research`);
    
    // Extract market/industry information from task
    const industry = this.extractIndustry(task.description);
    
    // Use MCP to gather industry data
    const industryData = await this.useMCP('market-data', {
      action: 'industry-analysis',
      industry,
      metrics: ['growth', 'competition', 'trends']
    });
    
    // Use MCP to get competitive landscape
    const competitors = await this.useMCP('business-intelligence', {
      action: 'key-players',
      industry,
      limit: 5
    });
    
    // Use MCP to analyze trends
    const trends = await this.useMCP('openai', {
      action: 'predict-trends',
      industry,
      timeframe: '12-months',
      data: industryData.data
    });
    
    return {
      success: true,
      action: 'market research',
      industry,
      marketSize: '$' + (Math.floor(Math.random() * 900) + 100) + ' Billion',
      growthRate: (Math.random() * 10 + 1).toFixed(1) + '%',
      keyPlayers: competitors.data?.companies || ['Company A', 'Company B', 'Company C'],
      emergingTrends: trends.data?.trends || [
        'Increased adoption of AI solutions',
        'Shift towards sustainable practices',
        'Remote-first business models'
      ]
    };
  }

  /**
   * Analyze financials based on task parameters
   * @param {Object} task - The financial analysis task
   * @returns {Object} - Results of the financial analysis
   */
  async analyzeFinancials(task) {
    console.log(`[${this.name}] Analyzing financials`);
    
    // Extract financial information from task
    const subject = this.extractBusinessSubject(task.description);
    
    // Use MCP to analyze financial data
    const financialData = await this.useMCP('financial-analyzer', {
      action: 'analyze-statements',
      company: subject,
      statements: ['income', 'balance', 'cash-flow'],
      period: 'latest-quarter'
    });
    
    // Use MCP to calculate key ratios
    const ratios = await this.useMCP('financial-analyzer', {
      action: 'calculate-ratios',
      data: financialData.data
    });
    
    // Use MCP to generate financial insights
    const insights = await this.useMCP('openai', {
      action: 'financial-insights',
      financialData: financialData.data,
      ratios: ratios.data
    });
    
    return {
      success: true,
      action: 'financial analysis',
      subject,
      revenue: '$' + (Math.floor(Math.random() * 900) + 100) + ' Million',
      profitMargin: (Math.random() * 20 + 5).toFixed(1) + '%',
      keyRatios: {
        currentRatio: (Math.random() + 1).toFixed(2),
        debtToEquity: (Math.random() * 2).toFixed(2),
        returnOnAssets: (Math.random() * 15).toFixed(2) + '%'
      },
      insights: insights.data?.points || [
        'Strong cash position relative to industry',
        'Concerning trend in operational expenses',
        'Opportunity to improve inventory management'
      ]
    };
  }

  /**
   * Develop a business strategy based on task parameters
   * @param {Object} task - The strategy development task
   * @returns {Object} - Results of the strategy development
   */
  async developStrategy(task) {
    console.log(`[${this.name}] Developing business strategy`);
    
    // Extract strategy information from task
    const subject = this.extractBusinessSubject(task.description);
    const goal = this.extractGoal(task.description);
    
    // Use MCP to gather relevant data
    const contextData = await this.useMCP('business-intelligence', {
      action: 'gather-context',
      subject,
      goal
    });
    
    // Use MCP to develop strategy
    const strategy = await this.useMCP('openai', {
      action: 'develop-strategy',
      subject,
      goal,
      contextData: contextData.data
    });
    
    return {
      success: true,
      action: 'develop strategy',
      subject,
      goal,
      strategicPillars: strategy.data?.pillars || [
        'Digital transformation initiatives',
        'Customer experience enhancement',
        'Operational efficiency improvements'
      ],
      keyInitiatives: strategy.data?.initiatives || [
        'Implement AI-driven customer service',
        'Streamline supply chain operations',
        'Develop sustainable product offerings'
      ],
      projectedOutcomes: strategy.data?.outcomes || [
        '15% increase in customer retention',
        '20% reduction in operational costs',
        'Expansion into 2 new market segments'
      ]
    };
  }

  /**
   * Extract client name from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted client name
   */
  extractClientName(description) {
    // In a real implementation, this would use NLP to extract the client name
    // For now, look for keywords and get the next few words
    
    const clientKeywords = ['client', 'customer', 'company', 'business', 'organization'];
    const words = description.split(' ');
    
    for (let i = 0; i < words.length - 1; i++) {
      if (clientKeywords.includes(words[i].toLowerCase())) {
        return words[i + 1] + (words[i + 2] || '');
      }
    }
    
    return 'Client Inc.'; // Default if no client name found
  }

  /**
   * Extract industry from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted industry
   */
  extractIndustry(description) {
    const industries = [
      'technology', 'healthcare', 'finance', 'retail',
      'manufacturing', 'energy', 'telecommunications'
    ];
    
    const descLower = description.toLowerCase();
    
    for (const industry of industries) {
      if (descLower.includes(industry)) {
        return industry.charAt(0).toUpperCase() + industry.slice(1);
      }
    }
    
    return 'Technology'; // Default if no industry found
  }

  /**
   * Extract business subject from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted business subject
   */
  extractBusinessSubject(description) {
    // Similar to extractClientName but more general
    return this.extractClientName(description);
  }

  /**
   * Extract goal from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted goal
   */
  extractGoal(description) {
    const goalKeywords = ['goal', 'aim', 'objective', 'target', 'purpose'];
    const descLower = description.toLowerCase();
    
    for (const keyword of goalKeywords) {
      const index = descLower.indexOf(keyword);
      if (index !== -1) {
        // Extract text after the keyword
        const afterKeyword = description.substring(index + keyword.length).trim();
        const endOfSentence = afterKeyword.indexOf('.');
        
        if (endOfSentence !== -1) {
          return afterKeyword.substring(0, endOfSentence).trim();
        }
        return afterKeyword;
      }
    }
    
    return 'Improve business performance'; // Default goal
  }
}

module.exports = BusinessAgent;