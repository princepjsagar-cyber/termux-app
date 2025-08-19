import os
import io
import base64
import asyncio
import logging
import json
import hashlib
import time
from typing import List, Dict, Any
from telegram import Update, InputFile, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
    MessageHandler,
    filters,
    InlineQueryHandler,
)

# Ultra-Advanced AI Agent Features
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import uuid

# Agent Memory and Context Management
class AgentMemory:
    def __init__(self):
        self.short_term = []  # Recent interactions
        self.long_term = {}   # Persistent knowledge
        self.episodic = []    # Event sequences
        self.semantic = {}    # Factual knowledge
        
    def add_interaction(self, user_id: str, message: str, response: str, context: dict = None):
        interaction = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'message': message,
            'response': response,
            'context': context or {},
            'type': 'interaction'
        }
        self.short_term.append(interaction)
        if len(self.short_term) > 50:  # Keep last 50 interactions
            self.short_term.pop(0)
    
    def add_knowledge(self, key: str, value: Any, category: str = 'general'):
        if category not in self.semantic:
            self.semantic[category] = {}
        self.semantic[category][key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.8
        }
    
    def get_relevant_context(self, query: str, limit: int = 5) -> List[dict]:
        # Simple relevance scoring (can be enhanced with embeddings)
        relevant = []
        for interaction in self.short_term[-20:]:  # Check last 20
            if any(word in interaction['message'].lower() for word in query.lower().split()):
                relevant.append(interaction)
        return relevant[:limit]

# Multi-Agent System
class AgentSystem:
    def __init__(self):
        self.agents = {
            'researcher': {'role': 'Research and fact-checking', 'active': True},
            'analyst': {'role': 'Data analysis and insights', 'active': True},
            'creative': {'role': 'Creative content generation', 'active': True},
            'planner': {'role': 'Task planning and execution', 'active': True},
            'moderator': {'role': 'Content moderation and safety', 'active': True}
        }
        self.memory = AgentMemory()
        self.conversation_threads = {}
        self.task_queue = []
        
    async def process_with_agents(self, user_id: str, message: str, context: dict = None) -> str:
        """Multi-agent processing pipeline"""
        # Create conversation thread
        if user_id not in self.conversation_threads:
            self.conversation_threads[user_id] = {
                'thread_id': str(uuid.uuid4()),
                'created': datetime.now().isoformat(),
                'messages': []
            }
        
        thread = self.conversation_threads[user_id]
        thread['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Agent processing pipeline
        tasks = []
        
        # 1. Moderator check
        if self.agents['moderator']['active']:
            tasks.append(self._moderate_content(message))
        
        # 2. Planner analysis
        if self.agents['planner']['active']:
            tasks.append(self._plan_response(message, context))
        
        # 3. Researcher fact-checking
        if self.agents['researcher']['active']:
            tasks.append(self._research_context(message))
        
        # 4. Analyst insights
        if self.agents['analyst']['active']:
            tasks.append(self._analyze_patterns(user_id, message))
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 5. Creative response generation
        creative_context = {
            'moderation': results[0] if len(results) > 0 else None,
            'plan': results[1] if len(results) > 1 else None,
            'research': results[2] if len(results) > 2 else None,
            'analysis': results[3] if len(results) > 3 else None,
            'memory': self.memory.get_relevant_context(message),
            'thread': thread
        }
        
        response = await self._generate_creative_response(message, creative_context)
        
        # Store in memory
        self.memory.add_interaction(user_id, message, response, creative_context)
        
        # Update thread
        thread['messages'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    async def _moderate_content(self, message: str) -> dict:
        """Content moderation agent"""
        # Simple keyword-based moderation (can be enhanced with AI)
        sensitive_words = ['spam', 'inappropriate', 'harmful']
        is_safe = not any(word in message.lower() for word in sensitive_words)
        
        return {
            'safe': is_safe,
            'confidence': 0.9,
            'flags': [] if is_safe else ['potential_concern'],
            'recommendation': 'proceed' if is_safe else 'review'
        }
    
    async def _plan_response(self, message: str, context: dict = None) -> dict:
        """Task planning agent"""
        # Analyze message intent and plan response strategy
        intent = self._detect_intent(message)
        
        plan = {
            'intent': intent,
            'strategy': self._get_strategy(intent),
            'steps': self._generate_steps(intent, message),
            'priority': 'high' if 'urgent' in message.lower() else 'normal',
            'estimated_tokens': len(message.split()) * 3
        }
        
        return plan
    
    async def _research_context(self, message: str) -> dict:
        """Research agent"""
        # Extract entities and facts for context
        entities = self._extract_entities(message)
        
        research = {
            'entities': entities,
            'facts': [],
            'sources': [],
            'confidence': 0.7
        }
        
        # Add to semantic memory
        for entity in entities:
            self.memory.add_knowledge(entity, {'mentioned_in': message}, 'entities')
        
        return research
    
    async def _analyze_patterns(self, user_id: str, message: str) -> dict:
        """Pattern analysis agent"""
        # Analyze user behavior patterns
        user_history = [i for i in self.memory.short_term if i['user_id'] == user_id]
        
        analysis = {
            'user_patterns': self._analyze_user_patterns(user_history),
            'conversation_flow': self._analyze_conversation_flow(user_history),
            'preferences': self._extract_preferences(user_history),
            'engagement_level': self._calculate_engagement(user_history)
        }
        
        return analysis
    
    async def _generate_creative_response(self, message: str, context: dict) -> str:
        """Creative response generation agent"""
        # Enhanced response generation with context
        base_prompt = f"""
        Context: {json.dumps(context, indent=2)}
        
        User message: {message}
        
        Generate a creative, helpful, and contextually aware response that:
        1. Addresses the user's intent
        2. Incorporates relevant context from memory
        3. Maintains conversation flow
        4. Provides value and insights
        5. Uses appropriate tone and style
        """
        
        # Use existing AI chat with enhanced context
        try:
            response = await self._call_ai_with_context(base_prompt, context)
            return response
        except Exception as e:
            logging.exception("Creative response generation failed: %s", e)
            return "I'm processing your request with advanced AI agents. Please try again."
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['help', 'support', 'assist']):
            return 'help_request'
        elif any(word in message_lower for word in ['explain', 'what is', 'how does']):
            return 'explanation_request'
        elif any(word in message_lower for word in ['create', 'generate', 'make']):
            return 'creation_request'
        elif any(word in message_lower for word in ['analyze', 'compare', 'evaluate']):
            return 'analysis_request'
        elif any(word in message_lower for word in ['search', 'find', 'look up']):
            return 'search_request'
        else:
            return 'general_conversation'
    
    def _get_strategy(self, intent: str) -> str:
        """Get response strategy based on intent"""
        strategies = {
            'help_request': 'provide_detailed_guidance',
            'explanation_request': 'give_clear_explanation_with_examples',
            'creation_request': 'guide_through_creation_process',
            'analysis_request': 'provide_structured_analysis',
            'search_request': 'conduct_comprehensive_search',
            'general_conversation': 'engage_naturally_with_context'
        }
        return strategies.get(intent, 'general_response')
    
    def _generate_steps(self, intent: str, message: str) -> List[str]:
        """Generate response steps"""
        if intent == 'help_request':
            return ['identify_specific_need', 'provide_step_by_step_guidance', 'offer_additional_resources']
        elif intent == 'explanation_request':
            return ['break_down_concept', 'provide_examples', 'connect_to_user_context']
        elif intent == 'creation_request':
            return ['clarify_requirements', 'suggest_approaches', 'guide_implementation']
        else:
            return ['understand_request', 'provide_response', 'follow_up']
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract named entities from message"""
        # Simple entity extraction (can be enhanced with NER)
        entities = []
        words = message.split()
        
        # Look for capitalized words, numbers, URLs, etc.
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
            elif word.startswith('http'):
                entities.append(word)
            elif word.isdigit():
                entities.append(word)
        
        return entities
    
    def _analyze_user_patterns(self, history: List[dict]) -> dict:
        """Analyze user behavior patterns"""
        if not history:
            return {'pattern': 'new_user', 'confidence': 0.5}
        
        patterns = {
            'message_frequency': len(history) / max(1, (datetime.now() - datetime.fromisoformat(history[0]['timestamp'])).days),
            'avg_message_length': sum(len(h['message'].split()) for h in history) / len(history),
            'preferred_topics': self._extract_topics(history),
            'interaction_style': self._classify_interaction_style(history)
        }
        
        return patterns
    
    def _analyze_conversation_flow(self, history: List[dict]) -> dict:
        """Analyze conversation flow patterns"""
        if len(history) < 2:
            return {'flow': 'initial', 'coherence': 0.5}
        
        flow_analysis = {
            'topic_consistency': self._calculate_topic_consistency(history),
            'response_time_patterns': self._analyze_response_times(history),
            'conversation_depth': self._calculate_conversation_depth(history),
            'engagement_trend': self._calculate_engagement_trend(history)
        }
        
        return flow_analysis
    
    def _extract_preferences(self, history: List[dict]) -> dict:
        """Extract user preferences from history"""
        preferences = {
            'response_length': 'medium',
            'technical_level': 'intermediate',
            'communication_style': 'friendly',
            'topics_of_interest': []
        }
        
        # Analyze message patterns to determine preferences
        avg_length = sum(len(h['message'].split()) for h in history) / len(history)
        if avg_length > 20:
            preferences['response_length'] = 'detailed'
        elif avg_length < 5:
            preferences['response_length'] = 'concise'
        
        return preferences
    
    def _calculate_engagement(self, history: List[dict]) -> float:
        """Calculate user engagement level"""
        if not history:
            return 0.5
        
        # Simple engagement scoring
        recent_messages = history[-10:]  # Last 10 messages
        engagement_score = 0.0
        
        for msg in recent_messages:
            # Longer messages = higher engagement
            engagement_score += min(len(msg['message'].split()) / 10, 1.0)
            
            # Questions = higher engagement
            if '?' in msg['message']:
                engagement_score += 0.2
            
            # Specific requests = higher engagement
            if any(word in msg['message'].lower() for word in ['please', 'can you', 'help']):
                engagement_score += 0.1
        
        return min(engagement_score / len(recent_messages), 1.0)
    
    def _extract_topics(self, history: List[dict]) -> List[str]:
        """Extract preferred topics from user history"""
        topics = []
        for msg in history:
            words = msg['message'].lower().split()
            # Simple topic extraction (can be enhanced)
            if any(word in words for word in ['ai', 'artificial', 'intelligence']):
                topics.append('AI/Technology')
            if any(word in words for word in ['news', 'current', 'events']):
                topics.append('News/Current Events')
            if any(word in words for word in ['code', 'programming', 'script']):
                topics.append('Programming/Code')
        
        return list(set(topics))
    
    def _classify_interaction_style(self, history: List[dict]) -> str:
        """Classify user interaction style"""
        if not history:
            return 'unknown'
        
        formal_count = sum(1 for msg in history if any(word in msg['message'].lower() for word in ['please', 'thank you', 'would you']))
        casual_count = sum(1 for msg in history if any(word in msg['message'].lower() for word in ['hey', 'hi', 'cool', 'awesome']))
        
        if formal_count > casual_count:
            return 'formal'
        elif casual_count > formal_count:
            return 'casual'
        else:
            return 'mixed'
    
    def _calculate_topic_consistency(self, history: List[dict]) -> float:
        """Calculate topic consistency across conversation"""
        if len(history) < 2:
            return 0.5
        
        # Simple topic consistency calculation
        topics = [self._extract_topics([msg]) for msg in history]
        consistent_topics = set.intersection(*[set(t) for t in topics if t])
        
        return len(consistent_topics) / max(1, len(set.union(*[set(t) for t in topics if t])))
    
    def _analyze_response_times(self, history: List[dict]) -> dict:
        """Analyze response time patterns"""
        if len(history) < 2:
            return {'avg_response_time': 0, 'pattern': 'unknown'}
        
        response_times = []
        for i in range(1, len(history)):
            prev_time = datetime.fromisoformat(history[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(history[i]['timestamp'])
            response_times.append((curr_time - prev_time).total_seconds())
        
        avg_response_time = sum(response_times) / len(response_times)
        
        return {
            'avg_response_time': avg_response_time,
            'pattern': 'fast' if avg_response_time < 60 else 'normal' if avg_response_time < 300 else 'slow'
        }
    
    def _calculate_conversation_depth(self, history: List[dict]) -> float:
        """Calculate conversation depth"""
        if len(history) < 2:
            return 0.5
        
        # Depth based on message complexity and follow-up questions
        depth_score = 0.0
        
        for msg in history:
            # Longer messages indicate depth
            depth_score += min(len(msg['message'].split()) / 20, 1.0)
            
            # Questions indicate depth
            if '?' in msg['message']:
                depth_score += 0.3
            
            # Technical terms indicate depth
            technical_terms = ['algorithm', 'function', 'api', 'database', 'framework']
            if any(term in msg['message'].lower() for term in technical_terms):
                depth_score += 0.2
        
        return min(depth_score / len(history), 1.0)
    
    def _calculate_engagement_trend(self, history: List[dict]) -> str:
        """Calculate engagement trend"""
        if len(history) < 5:
            return 'stable'
        
        # Compare recent vs older engagement
        recent = history[-5:]
        older = history[-10:-5] if len(history) >= 10 else history[:-5]
        
        recent_engagement = self._calculate_engagement(recent)
        older_engagement = self._calculate_engagement(older)
        
        if recent_engagement > older_engagement * 1.2:
            return 'increasing'
        elif recent_engagement < older_engagement * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _call_ai_with_context(self, prompt: str, context: dict) -> str:
        """Call AI with enhanced context"""
        # Use existing AI chat with enhanced prompt
        try:
            openai_key = _get_runtime_key(context.get('application', {}), 'OPENAI_API_KEY')
            if not openai_key:
                return "AI service not available. Please set OPENAI_API_KEY."
            
            client = openai.AsyncOpenAI(api_key=openai_key)
            
            # Create enhanced system message
            system_message = f"""
            You are an advanced AI agent with access to:
            - User conversation history and patterns
            - Contextual information and research
            - Multi-agent analysis results
            - Semantic memory and knowledge base
            
            Context: {json.dumps(context, indent=2)}
            
            Provide intelligent, contextually aware responses that:
            1. Address the user's specific intent and needs
            2. Incorporate relevant historical context
            3. Maintain natural conversation flow
            4. Provide actionable insights and value
            5. Adapt to user's communication style and preferences
            """
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.exception("AI call failed: %s", e)
            return "I'm experiencing technical difficulties. Please try again."

# Global agent system instance
agent_system = AgentSystem()

# Quantum-Inspired Advanced Features
import numpy as np
import hashlib
import secrets
from collections import defaultdict, deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Emotion Detection and Sentiment Analysis
class EmotionDetector:
    def __init__(self):
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'awesome', 'wonderful', 'amazing', 'love', '😊', '😄', '🎉'],
            'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'crying', '😢', '😭', '💔'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed', '😠', '😡', '💢'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'terrified', '😨', '😰', '😱'],
            'surprise': ['wow', 'omg', 'unexpected', 'shocked', '😲', '😱', '🤯'],
            'disgust': ['disgusting', 'gross', 'ew', 'yuck', '🤢', '🤮'],
            'neutral': ['okay', 'fine', 'normal', 'alright', '🤔', '😐']
        }
        self.emotion_history = defaultdict(list)
    
    def detect_emotion(self, text: str, user_id: str = None) -> dict:
        """Detect emotion from text"""
        text_lower = text.lower()
        emotion_scores = defaultdict(int)
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1
        
        # Analyze text patterns
        if '!' in text:
            emotion_scores['joy'] += 0.5
            emotion_scores['surprise'] += 0.3
        
        if '?' in text:
            emotion_scores['surprise'] += 0.3
        
        if text.isupper():
            emotion_scores['anger'] += 0.5
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[primary_emotion] / 3, 1.0)
        else:
            primary_emotion = 'neutral'
            confidence = 0.5
        
        result = {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_scores': dict(emotion_scores),
            'timestamp': datetime.now().isoformat()
        }
        
        if user_id:
            self.emotion_history[user_id].append(result)
            if len(self.emotion_history[user_id]) > 50:
                self.emotion_history[user_id].pop(0)
        
        return result

# Predictive Analytics Engine
class PredictiveEngine:
    def __init__(self):
        self.user_patterns = defaultdict(lambda: {
            'message_times': deque(maxlen=100),
            'topics': defaultdict(int),
            'response_times': deque(maxlen=50),
            'engagement_scores': deque(maxlen=50)
        })
        self.global_patterns = {
            'peak_hours': defaultdict(int),
            'popular_topics': defaultdict(int),
            'system_load': deque(maxlen=1000)
        }
    
    def predict_user_behavior(self, user_id: str) -> dict:
        """Predict user behavior patterns"""
        if user_id not in self.user_patterns:
            return {'prediction': 'new_user', 'confidence': 0.3}
        
        patterns = self.user_patterns[user_id]
        
        # Predict next message time
        if len(patterns['message_times']) > 2:
            time_diffs = []
            times = list(patterns['message_times'])
            for i in range(1, len(times)):
                diff = (times[i] - times[i-1]).total_seconds()
                time_diffs.append(diff)
            
            avg_interval = sum(time_diffs) / len(time_diffs)
            next_predicted = datetime.now() + timedelta(seconds=avg_interval)
        else:
            next_predicted = None
        
        # Predict preferred topics
        if patterns['topics']:
            preferred_topic = max(patterns['topics'], key=patterns['topics'].get)
        else:
            preferred_topic = 'general'
        
        return {
            'next_message_prediction': next_predicted.isoformat() if next_predicted else None,
            'preferred_topic': preferred_topic,
            'confidence': min(len(patterns['message_times']) / 20, 1.0)
        }

# Autonomous Learning System
class AutonomousLearner:
    def __init__(self):
        self.learning_modules = {
            'conversation_patterns': defaultdict(list),
            'user_preferences': defaultdict(dict),
            'system_optimizations': [],
            'error_patterns': defaultdict(int)
        }
        self.learning_rate = 0.1
    
    def learn_from_interaction(self, user_id: str, message: str, response: str, success: bool, context: dict):
        """Learn from each interaction"""
        pattern = {
            'message_type': self._classify_message(message),
            'response_type': self._classify_response(response),
            'success': success,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.learning_modules['conversation_patterns'][user_id].append(pattern)
        
        if not success:
            error_type = self._classify_error(message, response)
            self.learning_modules['error_patterns'][error_type] += 1
    
    def get_learning_insights(self) -> dict:
        """Get insights from learning system"""
        total_users = len(self.learning_modules['conversation_patterns'])
        total_interactions = sum(len(patterns) for patterns in self.learning_modules['conversation_patterns'].values())
        
        top_errors = sorted(self.learning_modules['error_patterns'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_users_learned': total_users,
            'total_interactions': total_interactions,
            'top_error_patterns': dict(top_errors),
            'learning_rate': self.learning_rate
        }
    
    def _classify_message(self, message: str) -> str:
        if '?' in message:
            return 'question'
        elif any(word in message.lower() for word in ['help', 'support', 'assist']):
            return 'help_request'
        else:
            return 'statement'
    
    def _classify_response(self, response: str) -> str:
        if len(response) > 200:
            return 'detailed'
        elif len(response) < 50:
            return 'concise'
        else:
            return 'standard'
    
    def _classify_error(self, message: str, response: str) -> str:
        if 'error' in response.lower():
            return 'system_error'
        else:
            return 'general_error'

# Real-time Collaboration System
class CollaborationSystem:
    def __init__(self):
        self.collaboration_sessions = {}
        self.shared_memory = defaultdict(dict)
        self.session_lock = threading.Lock()
    
    def create_session(self, session_id: str, participants: list, topic: str) -> dict:
        """Create a new collaboration session"""
        with self.session_lock:
            session = {
                'id': session_id,
                'participants': participants,
                'topic': topic,
                'created': datetime.now().isoformat(),
                'messages': [],
                'shared_knowledge': {},
                'status': 'active'
            }
            self.collaboration_sessions[session_id] = session
            return session

# Initialize advanced systems
emotion_detector = EmotionDetector()
predictive_engine = PredictiveEngine()
autonomous_learner = AutonomousLearner()
collaboration_system = CollaborationSystem()


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Neon Bot! I can help you with various tasks. "
        "Try /help to see what I can do."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lang = (getattr(user, "language_code", "") or "").lower()
    is_hi = lang.startswith("hi")

    if is_hi:
        user_help = (
            "उपलब्ध कमांड्स:\n\n"
            "मुख्य:\n"
            "/start — स्वागत संदेश\n"
            "/help, /commands — यह सहायता\n"
            "/status, /health — बॉट की स्थिति\n\n"
            "AI:\n"
            "/ai <प्रॉम्प्ट> — AI चैट (स्ट्रीमिंग)\n"
            "/ask <सवाल> — ताज़ा वेब संदर्भ + उत्तर\n"
            "/img <प्रॉम्प्ट> — इमेज जनरेशन (Gemini)\n\n"
            "खोज और समाचार:\n"
            "/web <क्वेरी> — वेब सर्च\n"
            "/news [SYMS] — ताज़ा सुर्खियाँ\n"
            "/subscribe_news [SYMS] [मिनट] — नियमित अपडेट\n"
            "/unsubscribe_news — अपडेट बंद\n"
            "/newsportal — न्यूज़ पोर्टल लिंक\n\n"
            "रेफरल:\n"
            "/referral — रेफरल API बेस URL\n\n"
            "Akka सर्वर:\n"
            "/akka — Akka-like सर्वर इंटरैक्शन\n\n"
            "🤖 AI एजेंट सिस्टम:\n"
            "/agent — मल्टी-एजेंट प्रोसेसिंग\n"
            "/memory — मेमोरी मैनेजमेंट\n"
            "/task — टास्क प्लानिंग और एक्जीक्यूशन\n"
            "/emotion — इमोशन डिटेक्शन\n"
            "/predict — प्रेडिक्टिव एनालिटिक्स\n"
            "/learn — ऑटोनोमस लर्निंग\n"
            "/collaborate — रीयल-टाइम कॉलैबोरेशन\n\n"
            "आवाज़ और विज़न:\n"
            "/tts <टेक्स्ट> — टेक्स्ट से आवाज़\n"
            "/ocr — इमेज से टेक्स्ट निकालें\n\n"
            "पर्सोना:\n"
            "/persona set <नाम> | /persona list — मोड बदलें\n"
        )
        admin_extra = (
            "\nएडमिन/कन्फ़िग:\n"
            "/feature <name> on|off — फीचर टॉगल\n"
            "/analytics — उपयोग काउंटर\n"
            "/setkey <NAME> <VALUE> — API की/कन्फ़िग सेट करें\n"
        )
    else:
        user_help = (
            "Available commands:\n\n"
            "Core:\n"
            "/start — Welcome message\n"
            "/help, /commands — This help\n"
            "/status, /health — Bot status & liveness\n\n"
            "AI:\n"
            "/ai <prompt> — Chat with AI (streaming)\n"
            "/ask <question> — Fresh web context + answer\n"
            "/img <prompt> — Image generation (Gemini)\n\n"
            "Search & News:\n"
            "/web <query> — Web search (CSE→Tavily→SerpAPI)\n"
            "/news [SYMS] — Latest headlines (e.g., AAPL,TSLA)\n"
            "/subscribe_news [SYMS] [minutes] — Push updates\n"
            "/unsubscribe_news — Stop pushes\n"
            "/newsportal — Portal link\n\n"
            "Referral:\n"
            "/referral — Referral API base URL\n\n"
            "Akka Server:\n"
            "/akka — Akka-like server interaction\n\n"
            "🤖 AI Agent System:\n"
            "/agent — Multi-agent processing\n"
            "/memory — Memory management\n"
            "/task — Task planning and execution\n"
            "/emotion — Emotion detection & sentiment analysis\n"
            "/predict — Predictive analytics & behavior prediction\n"
            "/learn — Autonomous learning system\n"
            "/collaborate — Real-time collaboration\n\n"
            "Voice & Vision:\n"
            "/tts <text> — Text to speech\n"
            "/ocr — Send/reply with an image to extract text\n\n"
            "Personalization:\n"
            "/persona set <name> | /persona list — Persona modes\n"
        )
        admin_extra = (
            "\nAdmin & Config:\n"
            "/feature <name> on|off — Toggle features\n"
            "/analytics — Usage counters\n"
            "/setkey <NAME> <VALUE> — Set API keys/config\n"
        )

    text = user_help + admin_extra
    await update.message.reply_text(text)


async def setkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Please provide a key: /setkey <name> <value>")
        return
    if len(args) < 2:
        await update.message.reply_text("Usage: /setkey <OPENAI_API_KEY|TAVILY_API_KEY|SERPAPI_KEY|GOOGLE_CSE_API_KEY|GOOGLE_CSE_CX|GEMINI_API_KEY|NEWS_API_KEY|NGROK_AUTHTOKEN|BOT_DATA_KEY> <value>")
        return
    name = args[0].strip().upper()
    value = " ".join(args[1:]).strip()
    allowed = {"OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX", "GEMINI_API_KEY", "NEWS_API_KEY", "NGROK_AUTHTOKEN", "BOT_DATA_KEY"}
    if name not in allowed:
        await update.message.reply_text("Key must be one of: " + ", ".join(sorted(allowed)))
        return
    context.application.bot_data[name] = value
    await update.message.reply_text("✅ Set " + name + " (in-memory)")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot Status:\n"
        f"• Token: {'Set' if BOT_TOKEN else 'Not Set'}\n"
        f"• Data Key: {'Set' if context.application.bot_data.get('BOT_DATA_KEY') else 'Not Set'}\n"
        f"• Store Path: {_get_store_path()}\n"
        f"• OPENAI_API_KEY: {'Set' if _get_runtime_key(context, 'OPENAI_API_KEY') else 'Not Set'}\n"
        f"• TAVILY_API_KEY: {'Set' if _get_runtime_key(context, 'TAVILY_API_KEY') else 'Not Set'}\n"
        f"• SERPAPI_KEY: {'Set' if _get_runtime_key(context, 'SERPAPI_KEY') else 'Not Set'}\n"
        f"• GOOGLE_CSE_API_KEY: {'Set' if _get_runtime_key(context, 'GOOGLE_CSE_API_KEY') else 'Not Set'}\n"
        f"• GOOGLE_CSE_CX: {'Set' if _get_runtime_key(context, 'GOOGLE_CSE_CX') else 'Not Set'}\n"
        f"• GEMINI_API_KEY: {'Set' if _get_runtime_key(context, 'GEMINI_API_KEY') else 'Not Set'}\n"
        f"• NEWS_API_KEY: {'Set' if _get_runtime_key(context, 'NEWS_API_KEY') else 'Not Set'}\n"
        f"• NGROK_AUTHTOKEN: {'Set' if _get_runtime_key(context, 'NGROK_AUTHTOKEN') else 'Not Set'}"
    )


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ai <prompt>")
        return
    await handle_ai_chat(update, context, prompt)


async def img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    gkey = get_gemini_key(context)
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /img <prompt>")
        return
    if not gkey:
        await update.message.reply_text("Set GEMINI_API_KEY with /setkey GEMINI_API_KEY <key>.")
        return

    try:
        import google.generativeai as genai
        genai.configure(api_key=gkey)
        model_name = os.environ.get("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-001")
        size = os.environ.get("GEMINI_IMAGE_SIZE", "1024x1024")
        # Try official image generation method first
        try:
            model = genai.GenerativeModel(model_name)
            result = model.generate_images(prompt=prompt, number_of_images=1, size=size)
        except Exception:
            # Fallback to generate_content for older SDKs
            model = genai.GenerativeModel(model_name)
            result = model.generate_content(prompt)

        b64 = None
        # Extract image base64 from various possible result shapes
        if hasattr(result, "generated_images") and result.generated_images:
            gi = result.generated_images[0]
            if hasattr(gi, "image"):
                b64 = getattr(gi.image, "base64_data", None) or getattr(gi.image, "data", None)
            b64 = b64 or getattr(gi, "base64_image", None)
        if not b64 and hasattr(result, "images") and result.images:
            img = result.images[0]
            b64 = getattr(img, "base64_data", None) or getattr(img, "data", None)
        if not b64 and hasattr(result, "candidates"):
            for cand in (result.candidates or []):
                for part in getattr(cand.content, "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        b64 = inline.data
                        break
                if b64:
                    break

        if not b64:
            await update.message.reply_text("❌ Gemini did not return an image.")
            return

        raw = base64.b64decode(b64)
        bio = io.BytesIO(raw)
        bio.name = "image.png"
        await update.message.reply_photo(photo=InputFile(bio, filename="image.png"), caption=f"🎨 {prompt[:200]}")
    except Exception as e:
        logging.exception("Gemini image generation error: %s", e)
        await update.message.reply_text("❌ Image generation failed.")


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    tc = get_tavily_client(context)
    serpapi_key = get_serpapi_key(context)
    gkey = get_google_cse_key(context)
    gcx = get_google_cse_cx(context)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /web <query>")
        return
    try:
        if gkey and gcx:
            params = {"key": gkey, "cx": gcx, "q": query, "num": 5}
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
                resp.raise_for_status()
                data = resp.json()
            items = data.get("items", [])
            lines = []
            for item in items[:5]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                lines.append(f"- {title}\n{snippet}\n{link}")
            body = "\n\n".join(lines) or "No results."
            await update.message.reply_text(f"🔎 {query}\n\n{body}"[:4096], disable_web_page_preview=True)
            return
        if tc:
            results = tc.search(query=query, search_depth="advanced", max_results=5)
            sources = results.get("results", [])
            summary = results.get("answer", "") or results.get("raw_content", "")
            if not summary:
                summary = "\n".join([s.get("content", "") for s in sources])[:1500]
            text = f"🔎 {query}\n\n{summary[:3500]}\n\n" + "\n".join([f"- {s.get('title','')}" for s in sources[:5]])
            await update.message.reply_text(text[:4096], disable_web_page_preview=True)
            return
        if serpapi_key:
            params = {"engine": "google", "q": query, "api_key": serpapi_key, "num": "5"}
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                resp.raise_for_status()
                data = resp.json()
            organic = data.get("organic_results", [])
            lines = []
            for item in organic[:5]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                lines.append(f"- {title}\n{snippet}\n{link}")
            body = "\n\n".join(lines) or "No results."
            await update.message.reply_text(f"🔎 {query}\n\n{body}"[:4096], disable_web_page_preview=True)
            return
        await update.message.reply_text("Set GOOGLE_CSE_API_KEY+GOOGLE_CSE_CX or TAVILY_API_KEY or SERPAPI_KEY to enable web search.")
    except Exception as e:
        logging.exception("Web search error: %s", e)
        await update.message.reply_text("❌ Web search failed.")


async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please provide a description for code generation.")
    context.user_data["code_description"] = update.message.text
    await update.message.reply_text("I'm ready to generate! Type /code again to continue.")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file_id = photo.file_id
    file = await context.bot.get_file(file_id)
    file_bytes = await file.download_as_bytearray()
    bio = io.BytesIO(file_bytes)
    bio.name = "photo.jpg"
    try:
        # Assuming a placeholder for image generation logic
        # In a real bot, you'd use a DALL-E client here
        await update.message.reply_text("🖼️ Image generation is not yet implemented.")
        # await update.message.reply_photo(InputFile(bio)) # Uncomment to send photo
    except Exception as e:
        logging.exception("Photo handler error: %s", e)
        await update.message.reply_text("❌ Failed to process photo.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Exception while handling update:", exc_info=context.error)
    if update and hasattr(update, "message") and update.message:
        try:
            await update.message.reply_text("⚠️ An error occurred. Please try again later.")
        except Exception:
            pass

# --------- Secure persistent store (encrypted, opt-in) ---------
def _get_store_path() -> str:
    return os.environ.get("BOT_STORE_PATH", "/workspace/.neon_store.enc")


def _get_data_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "BOT_DATA_KEY")


def _derive_fernet(key_str: str):
    from cryptography.fernet import Fernet
    # Derive a stable 32-byte key from any passphrase
    digest = hashlib.sha256(key_str.encode()).digest()
    fkey = base64.urlsafe_b64encode(digest)
    return Fernet(fkey)


def _ensure_store_root(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    app_data = context.application.bot_data
    if "STORE" not in app_data or not isinstance(app_data["STORE"], dict):
        app_data["STORE"] = {"users": {}}
    return app_data["STORE"]


def _get_user_entry(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Dict[str, Any]:
    store = _ensure_store_root(context)
    users = store.setdefault("users", {})
    entry = users.setdefault(str(user_id), {"items": [], "history": []})
    return entry


def load_store_eager(application) -> None:
    try:
        key = os.environ.get("BOT_DATA_KEY", "").strip()
        path = _get_store_path()
        store: Dict[str, Any] = {"users": {}}
        if key and os.path.exists(path):
            with open(path, "rb") as f:
                token = f.read()
            fernet = _derive_fernet(key)
            data = fernet.decrypt(token)
            store = json.loads(data.decode())
        application.bot_data["STORE"] = store
    except Exception as e:
        logging.exception("Failed to load store: %s", e)
        application.bot_data["STORE"] = {"users": {}}


def save_store(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        key = _get_data_key(context)
        if not key:
            return
        store = _ensure_store_root(context)
        data = json.dumps(store).encode()
        fernet = _derive_fernet(key)
        token = fernet.encrypt(data)
        path = _get_store_path()
        with open(path, "wb") as f:
            f.write(token)
    except Exception as e:
        logging.exception("Failed to save store: %s", e)


async def periodic_save_job(context: ContextTypes.DEFAULT_TYPE):
    save_store(context)


def _get_runtime_key(context: ContextTypes.DEFAULT_TYPE, name: str) -> str:
    try:
        app_data = context.application.bot_data
        val = app_data.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def append_user_message(history: List[Dict[str, Any]], text: str) -> None:
    history.append({"role": "user", "content": text})


def get_openai_client(context: ContextTypes.DEFAULT_TYPE):
    try:
        from openai import OpenAI
    except Exception:
        return None
    api_key = _get_runtime_key(context, "OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_tavily_client(context: ContextTypes.DEFAULT_TYPE):
    try:
        from tavily import TavilyClient
    except Exception:
        return None
    api_key = _get_runtime_key(context, "TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


def get_serpapi_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "SERPAPI_KEY")

def get_google_cse_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GOOGLE_CSE_API_KEY")

def get_google_cse_cx(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GOOGLE_CSE_CX")

def get_gemini_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GEMINI_API_KEY")

def get_news_api_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "NEWS_API_KEY")

def get_news_endpoint() -> str:
    return os.environ.get("NEWS_API_ENDPOINT", "https://api.example-news.io/v3/realtime").strip()

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    api_key = get_news_api_key(context)
    if not api_key:
        await update.message.reply_text("Set NEWS_API_KEY with /setkey NEWS_API_KEY <key>.")
        return
    # Symbols can be provided as args or from env; default AAPL,MSFT
    symbols_arg = "".join(context.args).strip() if context.args else ""
    if symbols_arg:
        symbols = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    else:
        symbols = [s.strip().upper() for s in os.environ.get("NEWS_SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
    limit = int(os.environ.get("NEWS_MAX_RESULTS", "8") or 8)
    params = {"symbols": ",".join(symbols), "limit": str(limit), "sort": "newest"}
    headers = {"X-Api-Key": api_key}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(get_news_endpoint(), params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # Accept both shapes: {articles: [...]} or {news: [...]} or direct list
        items = []
        if isinstance(data, dict):
            items = data.get("articles") or data.get("news") or []
        elif isinstance(data, list):
            items = data
        if not items:
            await update.message.reply_text("No news found.")
            return
        lines = []
        for it in items[:limit]:
            title = (it.get("title") if isinstance(it, dict) else str(it)) or "(untitled)"
            url = (it.get("url") if isinstance(it, dict) else "") or ""
            src = ""
            if isinstance(it, dict):
                src_obj = it.get("source") or {}
                if isinstance(src_obj, dict):
                    src = src_obj.get("name", "")
                elif isinstance(src_obj, str):
                    src = src_obj
            line = f"• {title}"
            if src:
                line += f" — {src}"
            if url:
                line += f"\n{url}"
            lines.append(line)
        text = "\n\n".join(lines)
        await update.message.reply_text(text[:4096], disable_web_page_preview=False)
    except Exception as e:
        logging.exception("News fetch error: %s", e)
        await update.message.reply_text("❌ Failed to fetch news.")


async def subscribe_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usage: /subscribe_news [AAPL,TSLA] [minutes]
    symbols = None
    interval_min = 30
    if context.args:
        parts = " ".join(context.args).split()
        if parts:
            if "," in parts[0] or parts[0].isalpha():
                symbols = [s.strip().upper() for s in parts[0].split(",") if s.strip()]
                parts = parts[1:]
        if parts:
            try:
                interval_min = max(5, int(parts[0]))
            except Exception:
                pass
    entry = _get_user_entry(context, update.effective_user.id)
    entry.setdefault("news_sub", {})
    if symbols:
        entry["news_sub"]["symbols"] = symbols
    entry["news_sub"]["interval_min"] = interval_min
    save_store(context)
    await update.message.reply_text(f"✅ Subscribed to news every {interval_min}m for symbols: {','.join(symbols or entry['news_sub'].get('symbols', ['AAPL','MSFT']))}")


async def unsubscribe_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry = _get_user_entry(context, update.effective_user.id)
    if "news_sub" in entry:
        entry.pop("news_sub", None)
        save_store(context)
        await update.message.reply_text("🛑 Unsubscribed from news updates.")
    else:
        await update.message.reply_text("You are not subscribed.")


async def _push_news_job(context: ContextTypes.DEFAULT_TYPE):
    import httpx
    app = context.application
    store = _ensure_store_root(context)
    api_key = _get_runtime_key(context, "NEWS_API_KEY")
    if not api_key:
        return
    for uid, udata in store.get("users", {}).items():
        sub = udata.get("news_sub")
        if not sub:
            continue
        symbols = sub.get("symbols") or [s.strip().upper() for s in os.environ.get("NEWS_SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
        limit = int(os.environ.get("NEWS_MAX_RESULTS", "5") or 5)
        params = {"symbols": ",".join(symbols), "limit": str(limit), "sort": "newest"}
        headers = {"X-Api-Key": api_key}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(get_news_endpoint(), params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            items = []
            if isinstance(data, dict):
                items = data.get("articles") or data.get("news") or []
            elif isinstance(data, list):
                items = data
            if not items:
                continue
            first = items[0]
            title = first.get("title", "(untitled)") if isinstance(first, dict) else str(first)
            url = (first.get("url") if isinstance(first, dict) else "") or ""
            msg = f"📰 Latest: {title}\n{url}" if url else f"📰 Latest: {title}"
            try:
                await app.bot.send_message(chat_id=int(uid), text=msg)
            except Exception:
                pass
        except Exception:
            continue


def ensure_history(context: ContextTypes.DEFAULT_TYPE) -> List[Dict[str, Any]]:
    user_data = context.user_data
    if "history" not in user_data:
        user_data["history"] = []
    # Trim to last 10 exchanges
    if len(user_data["history"]) > 20:
        user_data["history"] = user_data["history"][-20:]
    return user_data["history"]


def _aggregate_web_content(context: ContextTypes.DEFAULT_TYPE, query: str) -> str:
    import httpx
    gkey = get_google_cse_key(context)
    gcx = get_google_cse_cx(context)
    tc = get_tavily_client(context)
    serpapi_key = get_serpapi_key(context)
    try:
        if gkey and gcx:
            params = {"key": gkey, "cx": gcx, "q": query, "num": 5}
            with httpx.Client(timeout=15) as client:
                resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)
                resp.raise_for_status()
                data = resp.json()
            items = data.get("items", [])
            snippets = []
            for it in items[:5]:
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                link = it.get("link", "")
                snippets.append(f"{title}\n{snippet}\n{link}")
            return "\n\n".join(snippets)
        if tc:
            results = tc.search(query=query, search_depth="advanced", max_results=5)
            sources = results.get("results", [])
            if sources:
                return "\n\n".join([s.get("content", "") for s in sources[:5]])
        if serpapi_key:
            params = {"engine": "google", "q": query, "api_key": serpapi_key, "num": "5"}
            with httpx.Client(timeout=15) as client:
                resp = client.get("https://serpapi.com/search.json", params=params)
                resp.raise_for_status()
                data = resp.json()
            organic = data.get("organic_results", [])
            lines = []
            for it in organic[:5]:
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                link = it.get("link", "")
                lines.append(f"{title}\n{snippet}\n{link}")
            return "\n\n".join(lines)
    except Exception:
        pass
    return ""


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _rate_limit_ok(update.effective_user.id):
        return
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ask <question>")
        return
    if not _ask_quota_ok(context, update.effective_user.id):
        await update.message.reply_text("Daily /ask quota reached. Try tomorrow.")
        return
    flags = _get_flags(context)
    key = f"ask::{prompt.strip().lower()}"
    if flags.get("ask_cache"):
        cached = _cache_get(context, key)
        if cached:
            await update.message.reply_text(cached[:4096])
            _inc_usage(context, "ask_cache_hit")
            return
    fresh_context = _aggregate_web_content(context, prompt)
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY (and optionally web keys) to enable /ask.")
        return
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You answer with up-to-date info. Prefer the provided snippets; if unsure, say so."},
    ]
    if fresh_context:
        messages.append({"role": "system", "content": f"Snippets:\n{fresh_context}"})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if flags.get("ask_cache"):
            _cache_set(context, key, text)
        await update.message.reply_text(text[:4096])
        _inc_usage(context, "ask")
    except Exception as e:
        logging.exception("/ask error: %s", e)
        await update.message.reply_text("❌ Failed to answer.")

# Hook moderation and persona in AI chat
async def handle_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    if not _rate_limit_ok(update.effective_user.id):
        return
    if not await _moderate_if_enabled(context, prompt):
        await update.message.reply_text("❌ Content blocked by moderation.")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to enable AI chat.")
        return

    # Persona
    system_prompt = "You are a helpful assistant."
    try:
        persona = _get_user_entry(context, update.effective_user.id).get("persona")
        if persona == "tutor":
            system_prompt = "You are a patient tutor who explains step by step."
        elif persona == "coder":
            system_prompt = "You are a senior software engineer; respond with clear code-first solutions."
        elif persona == "analyst":
            system_prompt = "You are a data analyst; use bullet points and numbers."
        elif persona == "friendly":
            system_prompt = "You are friendly and concise."
    except Exception:
        pass

    try:
        user_id = update.effective_user.id
        entry = _get_user_entry(context, user_id)
        history = entry.setdefault("history", [])
        if len(history) > 20:
            entry["history"] = history[-20:]
            history = entry["history"]
    except Exception:
        history = ensure_history(context)

    # Prepend system
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": prompt}]

    placeholder = await update.message.reply_text("🤖 Thinking…")
    content_chunks: List[str] = []

    try:
        stream = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
            stream=True,
            temperature=0.4,
        )

        last_edit = ""
        async def periodic_edit():
            nonlocal last_edit
            while True:
                await asyncio.sleep(0.6)
                joined = "".join(content_chunks).strip()
                if joined and joined != last_edit:
                    try:
                        await placeholder.edit_text(joined[:4096])
                        last_edit = joined
                    except Exception:
                        pass

        edit_task = asyncio.create_task(periodic_edit())
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content_chunks.append(delta)
        edit_task.cancel()

        final_text = "".join(content_chunks).strip()
        entry = _get_user_entry(context, update.effective_user.id)
        hist = entry.setdefault("history", [])
        hist.append({"role": "user", "content": prompt})
        hist.append({"role": "assistant", "content": final_text})
        save_store(context)

        await placeholder.edit_text((final_text or "(no content)")[:4096])
        _inc_usage(context, "ai")
    except Exception as e:
        logging.exception("AI chat error: %s", e)
        try:
            await placeholder.edit_text("❌ AI error. Try again later.")
        except Exception:
            pass


async def add_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data
    args = context.args

    if not args:
        await update.message.reply_text("⚠️ Please provide data: /add <your_data>")
        return

    new_data = " ".join(args)
    if "items" not in user_data:
        user_data["items"] = []

    user_data["items"].append(new_data)

    # Also persist securely per user if BOT_DATA_KEY is set
    try:
        entry = _get_user_entry(context, update.effective_user.id)
        items = entry.setdefault("items", [])
        items.append(new_data)
        save_store(context)
    except Exception:
        pass

    await update.message.reply_text(f"✅ Added: {new_data}")


async def get_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Prefer encrypted store if available
    try:
        entry = _get_user_entry(context, update.effective_user.id)
        items = entry.get("items", [])
    except Exception:
        items = []

    if not items:
        user_data = context.user_data
        items = user_data.get("items", [])

    if not items:
        await update.message.reply_text("ℹ️ You haven't added any data yet!")
        return

    items_list = "\n".join(f"• {item}" for item in items)
    await update.message.reply_text(f"📦 Your stored data:\n{items_list}")


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client = get_openai_client(context)
    if not client:
        return
    voice = update.message.voice or update.message.audio
    if not voice:
        return
    tf = await context.bot.get_file(voice.file_id)
    # Download into memory
    file_bytes = await tf.download_as_bytearray()
    bio = io.BytesIO(file_bytes)
    bio.name = "audio.ogg"
    try:
        tr = client.audio.transcriptions.create(
            model=os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
            file=bio,
            response_format="text",
        )
        text = tr.strip() if isinstance(tr, str) else getattr(tr, "text", "").strip()
        if not text:
            text = "(empty transcription)"
        await update.message.reply_text(f"🗣️ {text[:4000]}")
    except Exception as e:
        logging.exception("Transcription error: %s", e)
        await update.message.reply_text("❌ Transcription failed.")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if update.message.text.strip().startswith("/"):
        return
    await handle_ai_chat(update, context, update.message.text.strip())


async def newsportal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    public_url_path = "/workspace/news_portal_url.txt"
    public_url = None
    try:
        if os.path.exists(public_url_path):
            with open(public_url_path, "r") as f:
                public_url = f.read().strip()
    except Exception:
        public_url = None
    if public_url:
        await update.message.reply_text(f"📰 News Portal: {public_url}")
        return
    host = os.environ.get("NEWS_PORTAL_HOST", "http://127.0.0.1:8080")
    await update.message.reply_text(f"📰 News Portal: {host}")


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query or ""
    if not query.strip():
        return
    # Simple AI-backed suggestion: echo + web hint
    suggestions = [
        InlineQueryResultArticle(
            id=str(int(time.time() * 1000)),
            title=f"Ask: {query[:50]}",
            input_message_content=InputTextMessageContent(f"/ask {query}")
        )
    ]
    try:
        await update.inline_query.answer(suggestions, cache_time=5, is_personal=True)
    except Exception:
        pass

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK")


async def referral_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    base = os.environ.get("REFERRAL_BASE", "http://127.0.0.1:8090")
    await update.message.reply_text(f"🎯 Referral API base: {base}\nPOST {base}/register (json: {{username}})\nPOST {base}/reward (json: {{referral_link}})")


async def akka_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interact with Akka-like server"""
    if not context.args:
        await update.message.reply_text(
            "Usage: /akka <command> [args]\n"
            "Commands:\n"
            "  message <text> - Send message to Akka server\n"
            "  state - Get current state\n"
            "  health - Check server health"
        )
        return
    
    command = context.args[0].lower()
    
    try:
        if command == "message" and len(context.args) > 1:
            text = " ".join(context.args[1:])
            async with httpx.AsyncClient() as client:
                response = await client.post(f"http://localhost:8070/message/{text}")
                if response.status_code == 200:
                    data = response.json()
                    await update.message.reply_text(f"✅ {data['response']}")
                else:
                    await update.message.reply_text("❌ Failed to process message")
        
        elif command == "state":
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8070/state")
                if response.status_code == 200:
                    data = response.json()
                    state = data['state']
                    await update.message.reply_text(
                        f"📊 Akka Server State:\n"
                        f"History size: {state['history_size']}\n"
                        f"Recent messages: {', '.join(state['history'][:3])}\n"
                        f"Users: {', '.join(state['user_data_keys'])}"
                    )
                else:
                    await update.message.reply_text("❌ Failed to get state")
        
        elif command == "health":
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8070/health")
                if response.status_code == 200:
                    data = response.json()
                    await update.message.reply_text(f"✅ {data['status']} - {data['service']}")
                else:
                    await update.message.reply_text("❌ Server not healthy")
        
        else:
            await update.message.reply_text("❌ Unknown command. Use /akka for help.")
    
    except Exception as e:
        logging.exception("Akka server error: %s", e)
        await update.message.reply_text("❌ Error connecting to Akka server")

async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Advanced AI agent system commands"""
    if not context.args:
        await update.message.reply_text(
            "🤖 **Advanced AI Agent System**\n\n"
            "**Commands:**\n"
            "• `/agent process <message>` - Multi-agent processing\n"
            "• `/agent memory` - View agent memory\n"
            "• `/agent status` - Agent system status\n"
            "• `/agent threads` - Active conversation threads\n"
            "• `/agent toggle <agent_name>` - Enable/disable agents\n"
            "• `/agent analyze <user_id>` - Analyze user patterns\n\n"
            "**Available Agents:**\n"
            "• `researcher` - Research and fact-checking\n"
            "• `analyst` - Data analysis and insights\n"
            "• `creative` - Creative content generation\n"
            "• `planner` - Task planning and execution\n"
            "• `moderator` - Content moderation and safety"
        )
        return
    
    command = context.args[0].lower()
    user_id = str(update.effective_user.id)
    
    try:
        if command == "process" and len(context.args) > 1:
            message = " ".join(context.args[1:])
            await update.message.reply_text("🤖 Processing with multi-agent system...")
            
            # Process with all agents
            response = await agent_system.process_with_agents(user_id, message, {
                'application': context.application.bot_data,
                'user_data': context.user_data
            })
            
            await update.message.reply_text(f"🤖 **Multi-Agent Response:**\n\n{response}")
        
        elif command == "memory":
            memory_info = {
                'short_term_count': len(agent_system.memory.short_term),
                'semantic_categories': list(agent_system.memory.semantic.keys()),
                'conversation_threads': len(agent_system.conversation_threads),
                'recent_interactions': [
                    {
                        'user_id': i['user_id'],
                        'message': i['message'][:50] + "..." if len(i['message']) > 50 else i['message'],
                        'timestamp': i['timestamp']
                    }
                    for i in agent_system.memory.short_term[-5:]
                ]
            }
            
            await update.message.reply_text(
                f"🧠 **Agent Memory Status:**\n\n"
                f"• Short-term interactions: {memory_info['short_term_count']}\n"
                f"• Semantic categories: {', '.join(memory_info['semantic_categories'])}\n"
                f"• Active threads: {memory_info['conversation_threads']}\n\n"
                f"**Recent Interactions:**\n" +
                "\n".join([
                    f"• {i['user_id']}: {i['message']} ({i['timestamp'][:19]})"
                    for i in memory_info['recent_interactions']
                ])
            )
        
        elif command == "status":
            active_agents = [name for name, config in agent_system.agents.items() if config['active']]
            inactive_agents = [name for name, config in agent_system.agents.items() if not config['active']]
            
            await update.message.reply_text(
                f"📊 **Agent System Status:**\n\n"
                f"✅ **Active Agents ({len(active_agents)}):**\n" +
                "\n".join([f"• {agent}" for agent in active_agents]) +
                f"\n\n❌ **Inactive Agents ({len(inactive_agents)}):**\n" +
                "\n".join([f"• {agent}" for agent in inactive_agents]) +
                f"\n\n🧠 Memory: {len(agent_system.memory.short_term)} interactions"
            )
        
        elif command == "threads":
            if not agent_system.conversation_threads:
                await update.message.reply_text("📝 No active conversation threads.")
                return
            
            threads_info = []
            for user_id, thread in agent_system.conversation_threads.items():
                threads_info.append(
                    f"• User {user_id}:\n"
                    f"  - Messages: {len(thread['messages'])}\n"
                    f"  - Created: {thread['created'][:19]}\n"
                    f"  - Thread ID: {thread['thread_id'][:8]}..."
                )
            
            await update.message.reply_text(
                f"📝 **Active Conversation Threads ({len(agent_system.conversation_threads)}):**\n\n" +
                "\n".join(threads_info)
            )
        
        elif command == "toggle" and len(context.args) > 1:
            agent_name = context.args[1].lower()
            if agent_name in agent_system.agents:
                agent_system.agents[agent_name]['active'] = not agent_system.agents[agent_name]['active']
                status = "✅ enabled" if agent_system.agents[agent_name]['active'] else "❌ disabled"
                await update.message.reply_text(f"🤖 Agent `{agent_name}` {status}")
            else:
                await update.message.reply_text(f"❌ Unknown agent: {agent_name}")
        
        elif command == "analyze" and len(context.args) > 1:
            target_user = context.args[1]
            user_history = [i for i in agent_system.memory.short_term if i['user_id'] == target_user]
            
            if not user_history:
                await update.message.reply_text(f"❌ No data found for user {target_user}")
                return
            
            # Analyze user patterns
            patterns = agent_system._analyze_user_patterns(user_history)
            flow = agent_system._analyze_conversation_flow(user_history)
            preferences = agent_system._extract_preferences(user_history)
            
            analysis_text = (
                f"🔍 **User Analysis: {target_user}**\n\n"
                f"**Patterns:**\n"
                f"• Message frequency: {patterns.get('message_frequency', 0):.2f}/day\n"
                f"• Avg message length: {patterns.get('avg_message_length', 0):.1f} words\n"
                f"• Preferred topics: {', '.join(patterns.get('preferred_topics', []))}\n"
                f"• Interaction style: {patterns.get('interaction_style', 'unknown')}\n\n"
                f"**Conversation Flow:**\n"
                f"• Topic consistency: {flow.get('topic_consistency', 0):.2f}\n"
                f"• Conversation depth: {flow.get('conversation_depth', 0):.2f}\n"
                f"• Engagement trend: {flow.get('engagement_trend', 'stable')}\n\n"
                f"**Preferences:**\n"
                f"• Response length: {preferences.get('response_length', 'medium')}\n"
                f"• Technical level: {preferences.get('technical_level', 'intermediate')}\n"
                f"• Communication style: {preferences.get('communication_style', 'friendly')}"
            )
            
            await update.message.reply_text(analysis_text)
        
        else:
            await update.message.reply_text("❌ Unknown agent command. Use `/agent` for help.")
    
    except Exception as e:
        logging.exception("Agent command error: %s", e)
        await update.message.reply_text("❌ Error in agent system. Please try again.")

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Memory management commands"""
    if not context.args:
        await update.message.reply_text(
            "🧠 **Memory Management**\n\n"
            "**Commands:**\n"
            "• `/memory add <key> <value>` - Add knowledge\n"
            "• `/memory get <key>` - Retrieve knowledge\n"
            "• `/memory list` - List all knowledge\n"
            "• `/memory clear` - Clear all memory\n"
            "• `/memory search <query>` - Search memory"
        )
        return
    
    command = context.args[0].lower()
    
    try:
        if command == "add" and len(context.args) > 2:
            key = context.args[1]
            value = " ".join(context.args[2:])
            agent_system.memory.add_knowledge(key, value, 'user_defined')
            await update.message.reply_text(f"✅ Added to memory: `{key}` = `{value}`")
        
        elif command == "get" and len(context.args) > 1:
            key = context.args[1]
            found = False
            for category, items in agent_system.memory.semantic.items():
                if key in items:
                    value = items[key]['value']
                    await update.message.reply_text(f"🧠 `{key}` = `{value}` (category: {category})")
                    found = True
                    break
            
            if not found:
                await update.message.reply_text(f"❌ Key `{key}` not found in memory")
        
        elif command == "list":
            if not agent_system.memory.semantic:
                await update.message.reply_text("🧠 Memory is empty")
                return
            
            memory_list = []
            for category, items in agent_system.memory.semantic.items():
                memory_list.append(f"**{category}:**")
                for key, data in items.items():
                    value = str(data['value'])
                    if len(value) > 50:
                        value = value[:50] + "..."
                    memory_list.append(f"• {key}: {value}")
                memory_list.append("")
            
            await update.message.reply_text("\n".join(memory_list))
        
        elif command == "clear":
            agent_system.memory.semantic.clear()
            agent_system.memory.short_term.clear()
            await update.message.reply_text("🧠 Memory cleared")
        
        elif command == "search" and len(context.args) > 1:
            query = " ".join(context.args[1:]).lower()
            results = []
            
            for category, items in agent_system.memory.semantic.items():
                for key, data in items.items():
                    if query in key.lower() or query in str(data['value']).lower():
                        value = str(data['value'])
                        if len(value) > 50:
                            value = value[:50] + "..."
                        results.append(f"• {key}: {value} (category: {category})")
            
            if results:
                await update.message.reply_text(f"🔍 **Search Results for '{query}':**\n\n" + "\n".join(results[:10]))
            else:
                await update.message.reply_text(f"🔍 No results found for '{query}'")
        
        else:
            await update.message.reply_text("❌ Unknown memory command. Use `/memory` for help.")
    
    except Exception as e:
        logging.exception("Memory command error: %s", e)
        await update.message.reply_text("❌ Error in memory system. Please try again.")

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Task planning and execution commands"""
    if not context.args:
        await update.message.reply_text(
            "📋 **Task Planning System**\n\n"
            "**Commands:**\n"
            "• `/task create <description>` - Create new task\n"
            "• `/task list` - List all tasks\n"
            "• `/task execute <task_id>` - Execute task\n"
            "• `/task status` - Task queue status\n"
            "• `/task clear` - Clear all tasks"
        )
        return
    
    command = context.args[0].lower()
    
    try:
        if command == "create" and len(context.args) > 1:
            description = " ".join(context.args[1:])
            task_id = str(uuid.uuid4())[:8]
            task = {
                'id': task_id,
                'description': description,
                'created': datetime.now().isoformat(),
                'status': 'pending',
                'user_id': str(update.effective_user.id)
            }
            agent_system.task_queue.append(task)
            await update.message.reply_text(f"📋 Task created: `{task_id}` - {description}")
        
        elif command == "list":
            if not agent_system.task_queue:
                await update.message.reply_text("📋 No tasks in queue")
                return
            
            task_list = []
            for task in agent_system.task_queue:
                status_emoji = "⏳" if task['status'] == 'pending' else "✅" if task['status'] == 'completed' else "❌"
                task_list.append(
                    f"{status_emoji} `{task['id']}` - {task['description']}\n"
                    f"   Status: {task['status']} | Created: {task['created'][:19]}"
                )
            
            await update.message.reply_text("📋 **Task Queue:**\n\n" + "\n\n".join(task_list))
        
        elif command == "execute" and len(context.args) > 1:
            task_id = context.args[1]
            task = next((t for t in agent_system.task_queue if t['id'] == task_id), None)
            
            if not task:
                await update.message.reply_text(f"❌ Task `{task_id}` not found")
                return
            
            if task['status'] != 'pending':
                await update.message.reply_text(f"❌ Task `{task_id}` is already {task['status']}")
                return
            
            # Execute task with agent system
            task['status'] = 'executing'
            await update.message.reply_text(f"🚀 Executing task: {task['description']}")
            
            # Process task with agents
            response = await agent_system.process_with_agents(
                task['user_id'], 
                f"Execute task: {task['description']}", 
                {'task_id': task_id}
            )
            
            task['status'] = 'completed'
            task['result'] = response
            await update.message.reply_text(f"✅ Task completed: {response}")
        
        elif command == "status":
            pending = len([t for t in agent_system.task_queue if t['status'] == 'pending'])
            completed = len([t for t in agent_system.task_queue if t['status'] == 'completed'])
            executing = len([t for t in agent_system.task_queue if t['status'] == 'executing'])
            
            await update.message.reply_text(
                f"📊 **Task Queue Status:**\n\n"
                f"⏳ Pending: {pending}\n"
                f"🚀 Executing: {executing}\n"
                f"✅ Completed: {completed}\n"
                f"📋 Total: {len(agent_system.task_queue)}"
            )
        
        elif command == "clear":
            agent_system.task_queue.clear()
            await update.message.reply_text("📋 Task queue cleared")
        
        else:
            await update.message.reply_text("❌ Unknown task command. Use `/task` for help.")
    
    except Exception as e:
        logging.exception("Task command error: %s", e)
        await update.message.reply_text("❌ Error in task system. Please try again.")

# Advanced config and helpers
OWNER_ID = int(os.environ.get("OWNER_ID", "0") or 0)

def _get_flags(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, bool]:
    flags = context.application.bot_data.setdefault("FEATURE_FLAGS", {})
    # defaults
    defaults = {
        "moderation": False,
        "tts": True,
        "ocr": True,
        "ask_cache": True,
        "quota": False,
    }
    for k, v in defaults.items():
        flags.setdefault(k, v)
    # Env overrides: FEATURE_<NAME>=on|off
    def env_on(name: str, default: bool) -> bool:
        val = os.environ.get(name)
        if not val:
            return default
        val = val.strip().lower()
        if val in ("1", "true", "on", "yes"): return True
        if val in ("0", "false", "off", "no"): return False
        return default
    flags["moderation"] = env_on("FEATURE_MODERATION", flags["moderation"]) 
    flags["quota"] = env_on("FEATURE_QUOTA", flags["quota"]) 
    flags["ask_cache"] = env_on("FEATURE_ASK_CACHE", flags["ask_cache"]) 
    flags["tts"] = env_on("FEATURE_TTS", flags["tts"]) 
    flags["ocr"] = env_on("FEATURE_OCR", flags["ocr"]) 
    return flags

# Simple per-user rate limiting
_last_msg_ts = {}

def _rate_limit_ok(user_id: int, min_interval_s: float = 0.5) -> bool:
    now = time.time()
    prev = _last_msg_ts.get(user_id, 0.0)
    if now - prev < min_interval_s:
        return False
    _last_msg_ts[user_id] = now
    return True

# Moderation
async def _moderate_if_enabled(context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    flags = _get_flags(context)
    if not flags.get("moderation"):
        return True
    try:
        client = get_openai_client(context)
        if not client:
            return True
        res = client.moderations.create(model=os.environ.get("OPENAI_MODERATION_MODEL", "omni-moderation-latest"), input=text)
        out = getattr(res, "results", [{}])[0]
        if out.get("flagged"):
            return False
    except Exception:
        return True
    return True

# Persona management
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if not args:
        entry = _get_user_entry(context, update.effective_user.id)
        persona = entry.get("persona") or "default"
        await update.message.reply_text(f"Current persona: {persona}\nUsage: /persona set <name> | /persona list")
        return
    if args[0].lower() == "list":
        await update.message.reply_text("Personas: default, tutor, coder, analyst, friendly")
        return
    if args[0].lower() == "set" and len(args) >= 2:
        name = args[1].strip().lower()
        entry = _get_user_entry(context, update.effective_user.id)
        entry["persona"] = name
        save_store(context)
        await update.message.reply_text(f"✅ Persona set: {name}")
        return
    await update.message.reply_text("Usage: /persona set <name> | /persona list")

# Analytics
def _inc_usage(context: ContextTypes.DEFAULT_TYPE, key: str):
    usage = context.application.bot_data.setdefault("USAGE", {})
    usage[key] = usage.get(key, 0) + 1

async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_ID and update.effective_user.id != OWNER_ID:
        await update.message.reply_text("Unauthorized")
        return
    usage = context.application.bot_data.get("USAGE", {})
    lines = [f"{k}: {v}" for k, v in sorted(usage.items())]
    await update.message.reply_text("Usage counts:\n" + ("\n".join(lines) or "(empty)"))

# Admin feature toggles
async def feature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_ID and update.effective_user.id != OWNER_ID:
        await update.message.reply_text("Unauthorized")
        return
    args = context.args or []
    flags = _get_flags(context)
    if len(args) < 2:
        await update.message.reply_text("Usage: /feature <name> on|off\n" + "\n".join([f"{k}={v}" for k, v in flags.items()]))
        return
    name, state = args[0], args[1].lower()
    if name not in flags:
        await update.message.reply_text("Unknown flag. Available: " + ", ".join(flags.keys()))
        return
    flags[name] = state == "on"
    await update.message.reply_text(f"✅ {name} set to {flags[name]}")

# Caching and quota for /ask
def _get_cache(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Dict[str, Any]]:
    return context.application.bot_data.setdefault("ASK_CACHE", {})

def _cache_get(context: ContextTypes.DEFAULT_TYPE, key: str, ttl_s: int = 600):
    cache = _get_cache(context)
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > ttl_s:
        cache.pop(key, None)
        return None
    return entry.get("val")

def _cache_set(context: ContextTypes.DEFAULT_TYPE, key: str, val: str):
    cache = _get_cache(context)
    cache[key] = {"ts": time.time(), "val": val}

# Daily quota tracking (per user)
from datetime import datetime

def _ask_quota_ok(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    flags = _get_flags(context)
    if not flags.get("quota"):
        return True
    limit = int(os.environ.get("DAILY_ASK_LIMIT", "50") or 50)
    entry = _get_user_entry(context, user_id)
    q = entry.setdefault("ask_quota", {})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if q.get("day") != today:
        q["day"], q["count"] = today, 0
    if q["count"] >= limit:
        return False
    q["count"] += 1
    save_store(context)
    return True

# TTS
async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flags = _get_flags(context)
    if not flags.get("tts"):
        await update.message.reply_text("TTS disabled")
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("Usage: /tts <text>")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to use TTS.")
        return
    try:
        speech = client.audio.speech.create(
            model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.environ.get("OPENAI_TTS_VOICE", "alloy"),
            input=text,
            format=os.environ.get("OPENAI_TTS_FORMAT", "mp3"),
        )
        audio_bytes = speech.read() if hasattr(speech, "read") else getattr(speech, "content", b"")
        if not audio_bytes:
            audio_bytes = speech  # some SDKs return bytes
        bio = io.BytesIO(audio_bytes)
        bio.name = "speech.mp3"
        await update.message.reply_voice(voice=InputFile(bio, filename="speech.mp3"))
    except Exception as e:
        logging.exception("TTS error: %s", e)
        await update.message.reply_text("❌ TTS failed.")

# OCR
async def ocr_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flags = _get_flags(context)
    if not flags.get("ocr"):
        await update.message.reply_text("OCR disabled")
        return
    if not update.message or not (update.message.photo or update.message.document):
        await update.message.reply_text("Reply to an image with /ocr or send an image with /ocr in caption.")
        return
    photo = None
    if update.message.photo:
        photo = update.message.photo[-1]
    elif update.message.document and str(update.message.document.mime_type or "").startswith("image/"):
        photo = update.message.document
    if not photo:
        await update.message.reply_text("No image found.")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to use OCR.")
        return
    tf = await context.bot.get_file(photo.file_id)
    image_url = tf.file_path
    messages = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract all visible text from this image."},
            {"type": "input_image", "image_url": image_url},
        ]
    }]
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        await update.message.reply_text(text[:4096])
    except Exception as e:
        logging.exception("OCR error: %s", e)
        await update.message.reply_text("❌ OCR failed.")

async def emotion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Emotion detection and sentiment analysis"""
    if not context.args:
        await update.message.reply_text(
            "😊 **Emotion Detection System**\n\n"
            "**Commands:**\n"
            "• `/emotion detect <text>` - Detect emotion in text\n"
            "• `/emotion history` - View your emotion history\n"
            "• `/emotion trend` - Analyze emotion trends\n"
            "• `/emotion analyze <user_id>` - Analyze user emotions"
        )
        return
    
    command = context.args[0].lower()
    user_id = str(update.effective_user.id)
    
    try:
        if command == "detect" and len(context.args) > 1:
            text = " ".join(context.args[1:])
            emotion_result = emotion_detector.detect_emotion(text, user_id)
            
            emoji_map = {
                'joy': '😊', 'sadness': '😢', 'anger': '😠', 
                'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
            }
            
            emoji = emoji_map.get(emotion_result['primary_emotion'], '😐')
            
            await update.message.reply_text(
                f"{emoji} **Emotion Analysis:**\n\n"
                f"**Primary Emotion:** {emotion_result['primary_emotion'].title()}\n"
                f"**Confidence:** {emotion_result['confidence']:.2f}\n"
                f"**All Emotions:**\n" +
                "\n".join([
                    f"• {emotion.title()}: {score}"
                    for emotion, score in emotion_result['all_scores'].items()
                    if score > 0
                ])
            )
        
        elif command == "history":
            if user_id not in emotion_detector.emotion_history:
                await update.message.reply_text("😐 No emotion history found.")
                return
            
            history = emotion_detector.emotion_history[user_id]
            if not history:
                await update.message.reply_text("😐 No emotion history found.")
                return
            
            # Analyze recent emotions
            recent_emotions = [h['primary_emotion'] for h in history[-10:]]
            emotion_counts = defaultdict(int)
            for emotion in recent_emotions:
                emotion_counts[emotion] += 1
            
            most_common = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
            
            await update.message.reply_text(
                f"📊 **Your Emotion History:**\n\n"
                f"**Recent Emotions:** {', '.join(recent_emotions)}\n"
                f"**Most Common:** {most_common.title()}\n"
                f"**Total Records:** {len(history)}\n"
                f"**Recent Analysis:** {history[-1]['primary_emotion'].title()} "
                f"({history[-1]['confidence']:.2f} confidence)"
            )
        
        elif command == "trend":
            if user_id not in emotion_detector.emotion_history:
                await update.message.reply_text("😐 No emotion data for trend analysis.")
                return
            
            history = emotion_detector.emotion_history[user_id]
            if len(history) < 5:
                await update.message.reply_text("😐 Need more emotion data for trend analysis.")
                return
            
            # Analyze trend
            recent = history[-5:]
            older = history[-10:-5] if len(history) >= 10 else history[:-5]
            
            recent_avg_confidence = sum(h['confidence'] for h in recent) / len(recent)
            older_avg_confidence = sum(h['confidence'] for h in older) / len(older)
            
            trend = "improving" if recent_avg_confidence > older_avg_confidence else "declining" if recent_avg_confidence < older_avg_confidence else "stable"
            
            await update.message.reply_text(
                f"📈 **Emotion Trend Analysis:**\n\n"
                f"**Trend:** {trend.title()}\n"
                f"**Recent Confidence:** {recent_avg_confidence:.2f}\n"
                f"**Previous Confidence:** {older_avg_confidence:.2f}\n"
                f"**Data Points:** {len(history)}"
            )
        
        elif command == "analyze" and len(context.args) > 1:
            target_user = context.args[1]
            if target_user not in emotion_detector.emotion_history:
                await update.message.reply_text(f"😐 No emotion data found for user {target_user}")
                return
            
            history = emotion_detector.emotion_history[target_user]
            emotions = [h['primary_emotion'] for h in history]
            emotion_counts = defaultdict(int)
            for emotion in emotions:
                emotion_counts[emotion] += 1
            
            most_common = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = sum(h['confidence'] for h in history) / len(history)
            
            await update.message.reply_text(
                f"🔍 **Emotion Analysis for {target_user}:**\n\n"
                f"**Most Common Emotion:** {most_common.title()}\n"
                f"**Average Confidence:** {avg_confidence:.2f}\n"
                f"**Total Records:** {len(history)}\n"
                f"**Emotion Distribution:**\n" +
                "\n".join([
                    f"• {emotion.title()}: {count}"
                    for emotion, count in emotion_counts.items()
                ])
            )
        
        else:
            await update.message.reply_text("❌ Unknown emotion command. Use `/emotion` for help.")
    
    except Exception as e:
        logging.exception("Emotion command error: %s", e)
        await update.message.reply_text("❌ Error in emotion detection. Please try again.")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Predictive analytics and behavior prediction"""
    if not context.args:
        await update.message.reply_text(
            "🔮 **Predictive Analytics System**\n\n"
            "**Commands:**\n"
            "• `/predict behavior <user_id>` - Predict user behavior\n"
            "• `/predict next <user_id>` - Predict next message time\n"
            "• `/predict topics <user_id>` - Predict preferred topics\n"
            "• `/predict system` - Predict system load"
        )
        return
    
    command = context.args[0].lower()
    
    try:
        if command == "behavior" and len(context.args) > 1:
            user_id = context.args[1]
            prediction = predictive_engine.predict_user_behavior(user_id)
            
            await update.message.reply_text(
                f"🔮 **Behavior Prediction for {user_id}:**\n\n"
                f"**Prediction Type:** {prediction['prediction']}\n"
                f"**Confidence:** {prediction['confidence']:.2f}\n"
                f"**Preferred Topic:** {prediction['preferred_topic']}\n"
                f"**Next Message:** {prediction['next_message_prediction'] or 'Unknown'}"
            )
        
        elif command == "next" and len(context.args) > 1:
            user_id = context.args[1]
            prediction = predictive_engine.predict_user_behavior(user_id)
            
            if prediction['next_message_prediction']:
                next_time = datetime.fromisoformat(prediction['next_message_prediction'])
                time_diff = next_time - datetime.now()
                
                if time_diff.total_seconds() > 0:
                    await update.message.reply_text(
                        f"⏰ **Next Message Prediction for {user_id}:**\n\n"
                        f"**Predicted Time:** {next_time.strftime('%H:%M:%S')}\n"
                        f"**Time Until:** {int(time_diff.total_seconds() / 60)} minutes\n"
                        f"**Confidence:** {prediction['confidence']:.2f}"
                    )
                else:
                    await update.message.reply_text(f"⏰ {user_id} is overdue for a message!")
            else:
                await update.message.reply_text(f"⏰ No prediction available for {user_id}")
        
        elif command == "topics" and len(context.args) > 1:
            user_id = context.args[1]
            prediction = predictive_engine.predict_user_behavior(user_id)
            
            await update.message.reply_text(
                f"📚 **Topic Prediction for {user_id}:**\n\n"
                f"**Preferred Topic:** {prediction['preferred_topic']}\n"
                f"**Confidence:** {prediction['confidence']:.2f}\n"
                f"**Prediction Quality:** {'High' if prediction['confidence'] > 0.7 else 'Medium' if prediction['confidence'] > 0.4 else 'Low'}"
            )
        
        elif command == "system":
            prediction = predictive_engine.predict_system_load()
            
            await update.message.reply_text(
                f"🖥️ **System Load Prediction:**\n\n"
                f"**Peak Hour:** {prediction['peak_hour']}:00\n"
                f"**Current Hour:** {prediction['current_hour']}:00\n"
                f"**Approaching Peak:** {'Yes' if prediction['approaching_peak'] else 'No'}\n"
                f"**Confidence:** {prediction['confidence']:.2f}"
            )
        
        else:
            await update.message.reply_text("❌ Unknown predict command. Use `/predict` for help.")
    
    except Exception as e:
        logging.exception("Predict command error: %s", e)
        await update.message.reply_text("❌ Error in predictive analytics. Please try again.")

async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Autonomous learning system commands"""
    if not context.args:
        await update_message.reply_text(
            "🧠 **Autonomous Learning System**\n\n"
            "**Commands:**\n"
            "• `/learn insights` - View learning insights\n"
            "• `/learn patterns` - View learned patterns\n"
            "• `/learn errors` - View error patterns\n"
            "• `/learn reset` - Reset learning data"
        )
        return
    
    command = context.args[0].lower()
    
    try:
        if command == "insights":
            insights = autonomous_learner.get_learning_insights()
            
            await update_message.reply_text(
                f"🧠 **Learning Insights:**\n\n"
                f"**Total Users Learned:** {insights['total_users_learned']}\n"
                f"**Total Interactions:** {insights['total_interactions']}\n"
                f"**Learning Rate:** {insights['learning_rate']:.2f}\n\n"
                f"**Top Error Patterns:**\n" +
                "\n".join([
                    f"• {error}: {count}"
                    for error, count in insights['top_error_patterns'].items()
                ])
            )
        
        elif command == "patterns":
            patterns = autonomous_learner.learning_modules['conversation_patterns']
            
            if not patterns:
                await update_message.reply_text("🧠 No conversation patterns learned yet.")
                return
            
            pattern_summary = []
            for user_id, user_patterns in list(patterns.items())[:5]:  # Show first 5 users
                if user_patterns:
                    success_rate = sum(1 for p in user_patterns if p['success']) / len(user_patterns)
                    pattern_summary.append(f"• User {user_id}: {len(user_patterns)} patterns, {success_rate:.2f} success rate")
            
            await update_message.reply_text(
                f"📊 **Learned Patterns:**\n\n" +
                "\n".join(pattern_summary) +
                f"\n\n**Total Users:** {len(patterns)}"
            )
        
        elif command == "errors":
            errors = autonomous_learner.learning_modules['error_patterns']
            
            if not errors:
                await update_message.reply_text("🧠 No error patterns recorded.")
                return
            
            error_list = []
            for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                error_list.append(f"• {error_type}: {count} occurrences")
            
            await update_message.reply_text(
                f"❌ **Error Patterns:**\n\n" +
                "\n".join(error_list)
            )
        
        elif command == "reset":
            autonomous_learner.learning_modules['conversation_patterns'].clear()
            autonomous_learner.learning_modules['error_patterns'].clear()
            await update_message.reply_text("🧠 Learning data reset successfully.")
        
        else:
            await update_message.reply_text("❌ Unknown learn command. Use `/learn` for help.")
    
    except Exception as e:
        logging.exception("Learn command error: %s", e)
        await update_message.reply_text("❌ Error in learning system. Please try again.")

async def collaborate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Real-time collaboration system"""
    if not context.args:
        await update_message.reply_text(
            "🤝 **Collaboration System**\n\n"
            "**Commands:**\n"
            "• `/collaborate create <topic>` - Create collaboration session\n"
            "• `/collaborate join <session_id>` - Join session\n"
            "• `/collaborate list` - List active sessions\n"
            "• `/collaborate summary <session_id>` - Session summary"
        )
        return
    
    command = context.args[0].lower()
    user_id = str(update.effective_user.id)
    
    try:
        if command == "create" and len(context.args) > 1:
            topic = " ".join(context.args[1:])
            session_id = str(uuid.uuid4())[:8]
            
            session = collaboration_system.create_session(session_id, [user_id], topic)
            
            await update_message.reply_text(
                f"🤝 **Collaboration Session Created:**\n\n"
                f"**Session ID:** `{session_id}`\n"
                f"**Topic:** {topic}\n"
                f"**Participants:** {len(session['participants'])}\n"
                f"**Status:** {session['status']}\n\n"
                f"Share this session ID with others to collaborate!"
            )
        
        elif command == "join" and len(context.args) > 1:
            session_id = context.args[1]
            
            if session_id not in collaboration_system.collaboration_sessions:
                await update_message.reply_text(f"❌ Session `{session_id}` not found.")
                return
            
            session = collaboration_system.collaboration_sessions[session_id]
            
            if user_id not in session['participants']:
                session['participants'].append(user_id)
            
            await update_message.reply_text(
                f"🤝 **Joined Collaboration Session:**\n\n"
                f"**Session ID:** `{session_id}`\n"
                f"**Topic:** {session['topic']}\n"
                f"**Participants:** {len(session['participants'])}\n"
                f"**Messages:** {len(session['messages'])}"
            )
        
        elif command == "list":
            sessions = collaboration_system.collaboration_sessions
            
            if not sessions:
                await update_message.reply_text("🤝 No active collaboration sessions.")
                return
            
            session_list = []
            for session_id, session in sessions.items():
                session_list.append(
                    f"• `{session_id}` - {session['topic']}\n"
                    f"  Participants: {len(session['participants'])}, "
                    f"Messages: {len(session['messages'])}"
                )
            
            await update_message.reply_text(
                f"🤝 **Active Collaboration Sessions:**\n\n" +
                "\n\n".join(session_list)
            )
        
        elif command == "summary" and len(context.args) > 1:
            session_id = context.args[1]
            
            if session_id not in collaboration_system.collaboration_sessions:
                await update_message.reply_text(f"❌ Session `{session_id}` not found.")
                return
            
            summary = collaboration_system.get_session_summary(session_id)
            
            await update_message.reply_text(
                f"📊 **Session Summary:**\n\n"
                f"**Session ID:** `{session_id}`\n"
                f"**Topic:** {summary['topic']}\n"
                f"**Participants:** {len(summary['participants'])}\n"
                f"**Messages:** {summary['message_count']}\n"
                f"**Duration:** {int(summary['duration'] / 60)} minutes\n"
                f"**Shared Knowledge:** {len(summary['shared_knowledge_keys'])} concepts\n"
                f"**Status:** {summary['status']}"
            )
        
        else:
            await update_message.reply_text("❌ Unknown collaborate command. Use `/collaborate` for help.")
    
    except Exception as e:
        logging.exception("Collaborate command error: %s", e)
        await update_message.reply_text("❌ Error in collaboration system. Please try again.")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set")

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .concurrent_updates(True)
        .build()
    )

    load_store_eager(application)
    try:
        application.job_queue.run_repeating(periodic_save_job, interval=60, first=60)
        application.job_queue.run_repeating(_push_news_job, interval=300, first=120)
    except Exception:
        pass

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("commands", help_command))
    application.add_handler(CommandHandler("add", add_data))
    application.add_handler(CommandHandler("get", get_data))
    application.add_handler(CommandHandler("setkey", setkey_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("img", img_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("newsportal", newsportal_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("subscribe_news", subscribe_news_command))
    application.add_handler(CommandHandler("unsubscribe_news", unsubscribe_news_command))
    application.add_handler(CommandHandler("code", code_command))
    application.add_handler(CommandHandler("health", health_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("analytics", analytics_command))
    application.add_handler(CommandHandler("feature", feature_command))
    application.add_handler(CommandHandler("tts", tts_command))
    application.add_handler(CommandHandler("ocr", ocr_command))
    application.add_handler(CommandHandler("referral", referral_command))
    application.add_handler(CommandHandler("akka", akka_command))
    application.add_handler(CommandHandler("agent", agent_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(CommandHandler("task", task_command))
    application.add_handler(CommandHandler("emotion", emotion_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("learn", learn_command))
    application.add_handler(CommandHandler("collaborate", collaborate_command))

    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_handler(InlineQueryHandler(inline_query_handler))

    application.add_error_handler(error_handler)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )

if __name__ == "__main__":
    main()