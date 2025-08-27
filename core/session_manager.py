#!/usr/bin/env python3
"""
Session Manager for Bit-Trade
Handles session persistence, resume capability, and state management
"""

import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

try:
    from .enhanced_logger import logger
except ImportError:
    from enhanced_logger import logger

@dataclass
class TradingSession:
    """Represents a trading session"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    mode: str  # demo, train, test
    episodes: int
    completed_episodes: int
    total_steps: int
    total_return: float
    win_rate: float
    config: Dict[str, Any]
    status: str  # active, paused, completed, failed
    last_checkpoint: Optional[str] = None

class SessionManager:
    """Manages trading sessions with persistence and resume capability"""
    
    def __init__(self, session_dir: str = "outputs/sessions"):
        self.session_dir = session_dir
        self.current_session: Optional[TradingSession] = None
        self.sessions_file = os.path.join(session_dir, "sessions_index.json")
        
        # Ensure session directory exists
        os.makedirs(session_dir, exist_ok=True)
        
        # Load existing sessions
        self.sessions_index = self._load_sessions_index()
        
        logger.info(f"📁 Session Manager initialized: {len(self.sessions_index)} previous sessions")
    
    def create_session(self, mode: str, config: Dict[str, Any]) -> str:
        """Create a new trading session"""
        session_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        session = TradingSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            mode=mode,
            episodes=config.get('episodes', 1),
            completed_episodes=0,
            total_steps=0,
            total_return=0.0,
            win_rate=0.0,
            config=config,
            status='active'
        )
        
        self.current_session = session
        self.sessions_index[session_id] = asdict(session)
        
        # Save session
        self._save_session(session)
        self._save_sessions_index()
        
        logger.info(f"🎯 New trading session created: {session_id}")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Episodes: {config.get('episodes', 1)}")
        logger.info(f"   Configuration: {config}")
        
        return session_id
    
    def update_session(self, **updates):
        """Update current session with new data"""
        if not self.current_session:
            logger.warning("⚠️ No active session to update")
            return
        
        # Update session fields
        for key, value in updates.items():
            if hasattr(self.current_session, key):
                setattr(self.current_session, key, value)
        
        # Update sessions index
        self.sessions_index[self.current_session.session_id] = asdict(self.current_session)
        
        # Save updates
        self._save_session(self.current_session)
        self._save_sessions_index()
        
        logger.debug(f"📊 Session updated: {self.current_session.session_id}")
    
    def checkpoint_session(self, agent_state: Dict, env_state: Dict):
        """Create a checkpoint for the current session"""
        if not self.current_session:
            logger.warning("⚠️ No active session to checkpoint")
            return
        
        checkpoint_time = datetime.now().isoformat()
        checkpoint_file = os.path.join(
            self.session_dir, 
            f"{self.current_session.session_id}_checkpoint_{checkpoint_time.replace(':', '-')}.pkl"
        )
        
        checkpoint_data = {
            'session_id': self.current_session.session_id,
            'checkpoint_time': checkpoint_time,
            'agent_state': agent_state,
            'env_state': env_state,
            'session_data': asdict(self.current_session)
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Update session with checkpoint info
            self.current_session.last_checkpoint = checkpoint_file
            self.update_session()
            
            logger.info(f"💾 Session checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def pause_session(self):
        """Pause the current session"""
        if not self.current_session:
            logger.warning("⚠️ No active session to pause")
            return
        
        self.current_session.status = 'paused'
        self.update_session()
        
        logger.info(f"⏸️ Session paused: {self.current_session.session_id}")
    
    def resume_session(self, session_id: str) -> Optional[Dict]:
        """Resume a paused session"""
        if session_id not in self.sessions_index:
            logger.error(f"❌ Session not found: {session_id}")
            return None
        
        session_data = self.sessions_index[session_id]
        
        if session_data['status'] not in ['paused', 'active']:
            logger.error(f"❌ Cannot resume session with status: {session_data['status']}")
            return None
        
        # Load session
        session = TradingSession(**session_data)
        self.current_session = session
        self.current_session.status = 'active'
        
        # Load checkpoint if available
        checkpoint_data = None
        if session.last_checkpoint and os.path.exists(session.last_checkpoint):
            try:
                with open(session.last_checkpoint, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                logger.info(f"📂 Checkpoint loaded: {session.last_checkpoint}")
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
        
        self.update_session()
        logger.info(f"▶️ Session resumed: {session_id}")
        
        return checkpoint_data
    
    def complete_session(self, final_results: Dict):
        """Mark session as completed with final results"""
        if not self.current_session:
            logger.warning("⚠️ No active session to complete")
            return
        
        self.current_session.status = 'completed'
        self.current_session.end_time = datetime.now().isoformat()
        
        # Update final statistics
        if 'total_return' in final_results:
            self.current_session.total_return = final_results['total_return']
        if 'win_rate' in final_results:
            self.current_session.win_rate = final_results['win_rate']
        if 'completed_episodes' in final_results:
            self.current_session.completed_episodes = final_results['completed_episodes']
        
        self.update_session()
        
        # Save final results
        results_file = os.path.join(self.session_dir, f"{self.current_session.session_id}_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'session': asdict(self.current_session),
                    'results': final_results
                }, f, indent=2, default=str)
            
            logger.info(f"✅ Session completed: {self.current_session.session_id}")
            logger.info(f"   Duration: {self._calculate_duration()}")
            logger.info(f"   Episodes: {self.current_session.completed_episodes}/{self.current_session.episodes}")
            logger.info(f"   Total Return: {self.current_session.total_return:.2%}")
            logger.info(f"   Results saved: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save session results: {e}")
    
    def fail_session(self, error_message: str):
        """Mark session as failed"""
        if not self.current_session:
            return
        
        self.current_session.status = 'failed'
        self.current_session.end_time = datetime.now().isoformat()
        
        # Save error details
        error_file = os.path.join(self.session_dir, f"{self.current_session.session_id}_error.json")
        try:
            with open(error_file, 'w') as f:
                json.dump({
                    'session_id': self.current_session.session_id,
                    'error_message': error_message,
                    'error_time': datetime.now().isoformat(),
                    'session_data': asdict(self.current_session)
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save error details: {e}")
        
        self.update_session()
        logger.error(f"❌ Session failed: {self.current_session.session_id} - {error_message}")
    
    def list_sessions(self, status: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """List available sessions"""
        sessions = list(self.sessions_index.values())
        
        # Filter by status if specified
        if status:
            sessions = [s for s in sessions if s.get('status') == status]
        
        # Sort by start time (most recent first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return sessions[:limit]
    
    def get_session_stats(self) -> Dict:
        """Get overall session statistics"""
        if not self.sessions_index:
            return {
                "total_sessions": 0,
                "completed_sessions": 0,
                "active_sessions": 0,
                "failed_sessions": 0,
                "avg_return": 0.0,
                "total_episodes": 0
            }
        
        sessions = list(self.sessions_index.values())
        completed = [s for s in sessions if s.get('status') == 'completed']
        
        stats = {
            "total_sessions": len(sessions),
            "completed_sessions": len(completed),
            "active_sessions": len([s for s in sessions if s.get('status') in ['active', 'paused']]),
            "failed_sessions": len([s for s in sessions if s.get('status') == 'failed']),
            "total_episodes": sum(s.get('completed_episodes', 0) for s in sessions)
        }
        
        if completed:
            returns = [s.get('total_return', 0) for s in completed]
            stats["avg_return"] = sum(returns) / len(returns)
            stats["best_return"] = max(returns)
            stats["worst_return"] = min(returns)
        else:
            stats["avg_return"] = 0.0
            stats["best_return"] = 0.0
            stats["worst_return"] = 0.0
        
        return stats
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up old session files"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for session_id, session_data in list(self.sessions_index.items()):
            try:
                session_start = datetime.fromisoformat(session_data.get('start_time', ''))
                
                if session_start < cutoff_date and session_data.get('status') in ['completed', 'failed']:
                    # Remove session files
                    session_files = [
                        os.path.join(self.session_dir, f"{session_id}.json"),
                        os.path.join(self.session_dir, f"{session_id}_results.json"),
                        os.path.join(self.session_dir, f"{session_id}_error.json")
                    ]
                    
                    # Remove checkpoint files
                    for file in os.listdir(self.session_dir):
                        if file.startswith(f"{session_id}_checkpoint_"):
                            session_files.append(os.path.join(self.session_dir, file))
                    
                    for file_path in session_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # Remove from index
                    del self.sessions_index[session_id]
                    cleaned_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to cleanup session {session_id}: {e}")
        
        if cleaned_count > 0:
            self._save_sessions_index()
            logger.info(f"🧹 Cleaned up {cleaned_count} old sessions (older than {days} days)")
    
    def _load_sessions_index(self) -> Dict:
        """Load sessions index from file"""
        if not os.path.exists(self.sessions_file):
            return {}
        
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load sessions index: {e}")
            return {}
    
    def _save_sessions_index(self):
        """Save sessions index to file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save sessions index: {e}")
    
    def _save_session(self, session: TradingSession):
        """Save individual session to file"""
        session_file = os.path.join(self.session_dir, f"{session.session_id}.json")
        
        try:
            with open(session_file, 'w') as f:
                json.dump(asdict(session), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save session {session.session_id}: {e}")
    
    def _calculate_duration(self) -> str:
        """Calculate session duration"""
        if not self.current_session or not self.current_session.end_time:
            return "Unknown"
        
        start = datetime.fromisoformat(self.current_session.start_time)
        end = datetime.fromisoformat(self.current_session.end_time)
        duration = end - start
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

if __name__ == "__main__":
    # Test session manager
    manager = SessionManager()
    
    # Test creating a session
    config = {"episodes": 5, "mcts_sims": 10, "symbols": ["BTCUSDT"]}
    session_id = manager.create_session("demo", config)
    
    # Test updating session
    manager.update_session(completed_episodes=2, total_return=0.05)
    
    # Test checkpoint
    agent_state = {"experience_buffer_size": 50}
    env_state = {"portfolio_value": 10500}
    manager.checkpoint_session(agent_state, env_state)
    
    # Test completion
    final_results = {"total_return": 0.08, "win_rate": 0.6, "completed_episodes": 5}
    manager.complete_session(final_results)
    
    # Test statistics
    stats = manager.get_session_stats()
    print(f"📊 Session statistics: {stats}")
    
    print("✅ Session Manager test completed")