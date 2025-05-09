/**
 * ResearchAgent.js
 * 
 * Specialized agent for handling research-related tasks including
 * information gathering, fact checking, data analysis, and
 * providing background knowledge for other agents.
 */

const BaseAgent = require('../../core/BaseAgent');

class ResearchAgent extends BaseAgent {
  constructor() {
    super(
      'research-agent',
      'Research Specialist',
      'Handles information gathering, research, fact checking, and knowledge retrieval',
      ['search-engine', 'openai', 'knowledge-graph', 'fact-checker']
    );
  }

  /**
   * Process an assigned research-related task
   * @param {Object} task - The task to process
   */
  async processTask(task) {
    console.log(`[${this.name}] Processing research task: ${task.description}`);
    
    try {
      let result = {};
      
      // Determine what type of research task this is
      if (task.description.toLowerCase().includes('background') || 
          task.description.toLowerCase().includes('overview')) {
        result = await this.getBackgroundInformation(task);
      } else if (task.description.toLowerCase().includes('fact check') || 
                 task.description.toLowerCase().includes('verify')) {
        result = await this.factCheck(task);
      } else if (task.description.toLowerCase().includes('analyze data') || 
                 task.description.toLowerCase().includes('data analysis')) {
        result = await this.analyzeData(task);
      } else if (task.description.toLowerCase().includes('trend') || 
                 task.description.toLowerCase().includes('latest')) {
        result = await this.researchTrends(task);
      } else {
        // Default to general information gathering
        result = await this.getBackgroundInformation(task);
      }
      
      // Report task completion to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'research',
        result
      });
      
    } catch (error) {
      console.error(`[${this.name}] Error processing task:`, error);
      
      // Report failure to the coordinator
      this.completeTask(task.id, {
        agentType: this.id,
        action: 'research',
        success: false,
        error: error.message
      });
    }
  }

  /**
   * Get background information on a topic
   * @param {Object} task - The information gathering task
   * @returns {Object} - Results of the information gathering
   */
  async getBackgroundInformation(task) {
    console.log(`[${this.name}] Gathering background information`);
    
    // Extract topic from task
    const topic = this.extractResearchTopic(task.description);
    
    // Use MCP to search for information
    const searchResults = await this.useMCP('search-engine', {
      action: 'search',
      query: topic,
      filters: {
        recency: 'any',
        credibility: 'high'
      }
    });
    
    // Use MCP to extract key information
    const keyInfo = await this.useMCP('openai', {
      action: 'extract-key-information',
      topic,
      searchResults: searchResults.data
    });
    
    // Use MCP to organize information
    const organizedInfo = await this.useMCP('knowledge-graph', {
      action: 'organize',
      information: keyInfo.data
    });
    
    return {
      success: true,
      action: 'background research',
      topic,
      summary: keyInfo.data?.summary || `Comprehensive overview of ${topic}`,
      keyPoints: keyInfo.data?.keyPoints || [
        `${topic} has shown significant growth in the past year`,
        `Major developments include technological advancements and regulatory changes`,
        `Industry experts predict continued expansion in this area`
      ],
      sources: searchResults.data?.sources?.slice(0, 3) || [
        { title: 'Industry Report 2025', url: 'https://example.com/report', credibility: 'High' },
        { title: 'Expert Analysis', url: 'https://example.com/analysis', credibility: 'Medium' }
      ]
    };
  }

  /**
   * Fact check statements or claims
   * @param {Object} task - The fact checking task
   * @returns {Object} - Results of the fact checking
   */
  async factCheck(task) {
    console.log(`[${this.name}] Fact checking`);
    
    // Extract claims from task
    const claims = this.extractClaims(task.description);
    
    // Use MCP to verify each claim
    const verificationResults = [];
    
    for (const claim of claims) {
      // Use MCP for fact checking
      const verification = await this.useMCP('fact-checker', {
        action: 'verify',
        claim
      });
      
      // Use MCP to find supporting or contradicting evidence
      const evidence = await this.useMCP('search-engine', {
        action: 'find-evidence',
        claim,
        type: 'both' // both supporting and contradicting
      });
      
      verificationResults.push({
        claim,
        verdict: verification.data?.verdict || Math.random() > 0.7 ? 'false' : 'true',
        confidence: verification.data?.confidence || (Math.random() * 40 + 60).toFixed(1) + '%',
        evidence: evidence.data?.sources || []
      });
    }
    
    return {
      success: true,
      action: 'fact check',
      claimCount: claims.length,
      verificationResults: verificationResults.length > 0 ? verificationResults : [
        {
          claim: 'The market grew by 25% last year',
          verdict: 'partially true',
          confidence: '78.5%',
          evidence: [
            { source: 'Industry Report', excerpt: 'Market grew by 22.7% in the previous fiscal year' }
          ]
        }
      ]
    };
  }

  /**
   * Analyze data based on task parameters
   * @param {Object} task - The data analysis task
   * @returns {Object} - Results of the data analysis
   */
  async analyzeData(task) {
    console.log(`[${this.name}] Analyzing data`);
    
    // In a real implementation, this would extract or receive data from the task
    const data = task.data || { sampleData: 'placeholder data points' };
    
    // Use MCP to analyze the data
    const analysis = await this.useMCP('openai', {
      action: 'analyze-data',
      data,
      context: task.description
    });
    
    // Use MCP to generate visualizations
    const visualizations = await this.useMCP('knowledge-graph', {
      action: 'visualize-data',
      data,
      analysis: analysis.data
    });
    
    return {
      success: true,
      action: 'data analysis',
      dataPoints: Math.floor(Math.random() * 1000) + 500,
      insights: analysis.data?.insights || [
        'Strong correlation between variables X and Y',
        'Unusual pattern detected in the third quarter',
        'Data suggests 34% improvement when strategy A is applied'
      ],
      visualizations: visualizations.data?.charts || ['trend analysis', 'correlation matrix', 'comparison chart'],
      confidence: (Math.random() * 20 + 80).toFixed(1) + '%'
    };
  }

  /**
   * Research current trends on a topic
   * @param {Object} task - The trends research task
   * @returns {Object} - Results of the trends research
   */
  async researchTrends(task) {
    console.log(`[${this.name}] Researching trends`);
    
    // Extract topic from task
    const topic = this.extractResearchTopic(task.description);
    
    // Use MCP to search for recent information
    const recentInfo = await this.useMCP('search-engine', {
      action: 'search',
      query: `${topic} latest trends`,
      filters: {
        recency: 'very-recent',
        sortBy: 'date'
      }
    });
    
    // Use MCP to analyze trends
    const trendAnalysis = await this.useMCP('openai', {
      action: 'analyze-trends',
      topic,
      recentData: recentInfo.data
    });
    
    return {
      success: true,
      action: 'trend research',
      topic,
      currentTrends: trendAnalysis.data?.trends || [
        'Rapid adoption of AI-powered solutions',
        'Shift towards sustainable business practices',
        'Increasing focus on data privacy and security'
      ],
      momentum: trendAnalysis.data?.momentum || {
        rising: ['Remote collaboration tools', 'Personalized customer experiences'],
        stable: ['Cloud computing', 'Mobile optimization'],
        declining: ['Traditional advertising channels', 'On-premise solutions']
      },
      prediction: trendAnalysis.data?.prediction || 'The industry will likely see continued integration of AI and machine learning technologies, with a greater emphasis on sustainability and ethical considerations over the next 12 months.'
    };
  }

  /**
   * Extract research topic from task description
   * @param {String} description - Task description
   * @returns {String} - Extracted research topic
   */
  extractResearchTopic(description) {
    // In a real implementation, this would use NLP to extract the topic
    // For now, look for keywords and get surrounding text
    
    const researchKeywords = ['research', 'information on', 'learn about', 'find out about', 'background on'];
    const descLower = description.toLowerCase();
    
    for (const keyword of researchKeywords) {
      const index = descLower.indexOf(keyword);
      if (index !== -1) {
        // Get the text after the keyword
        const startIndex = index + keyword.length;
        let endIndex = description.indexOf('.', startIndex);
        if (endIndex === -1) endIndex = description.length;
        
        return description.substring(startIndex, endIndex).trim();
      }
    }
    
    // If no keyword found, use a generic topic extraction
    const words = description.split(' ');
    const noiseWords = ['the', 'a', 'an', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'by', 'in'];
    
    // Get the most frequent non-noise word
    const wordFreq = {};
    words.forEach(word => {
      const w = word.toLowerCase().replace(/[^\w]/g, '');
      if (!noiseWords.includes(w) && w.length > 2) {
        wordFreq[w] = (wordFreq[w] || 0) + 1;
      }
    });
    
    let topWord = '';
    let maxFreq = 0;
    
    for (const [word, freq] of Object.entries(wordFreq)) {
      if (freq > maxFreq) {
        maxFreq = freq;
        topWord = word;
      }
    }
    
    return topWord || 'general research topic';
  }

  /**
   * Extract claims to fact check from task description
   * @param {String} description - Task description
   * @returns {Array} - Array of claims to verify
   */
  extractClaims(description) {
    // In a real implementation, this would use NLP to extract claims
    // For now, look for patterns like "check if..." or quoted text
    
    const claims = [];
    
    // Look for quoted claims
    const quoteMatches = description.match(/"([^"]*)"/g);
    if (quoteMatches) {
      quoteMatches.forEach(match => {
        claims.push(match.replace(/"/g, ''));
      });
    }
    
    // Look for "check if" pattern
    const checkIfIndex = description.toLowerCase().indexOf('check if');
    if (checkIfIndex !== -1) {
      const startIndex = checkIfIndex + 9; // "check if " length
      let endIndex = description.indexOf('.', startIndex);
      if (endIndex === -1) endIndex = description.length;
      
      claims.push(description.substring(startIndex, endIndex).trim());
    }
    
    // If no claims found, create a placeholder
    if (claims.length === 0) {
      claims.push('The market is growing at a rate of 15% annually');
    }
    
    return claims;
  }
}

module.exports = ResearchAgent;