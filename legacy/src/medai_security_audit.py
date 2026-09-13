# security_audit.py - Sistema de segurança, criptografia e auditoria

import hashlib
import secrets
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import sqlite3
import bcrypt
from enum import Enum
import re
import os
import base64
from functools import wraps
import threading
import time

logger = logging.getLogger('MedAI.Security')

class UserRole(Enum):
    """Papéis de usuário no sistema"""
    ADMIN = "admin"
    PHYSICIAN = "physician"
    RADIOLOGIST = "radiologist"
    TECHNICIAN = "technician"
    VIEWER = "viewer"
    SYSTEM = "system"

class AuditEventType(Enum):
    """Tipos de eventos de auditoria"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    VIEW_IMAGE = "view_image"
    ANALYZE_IMAGE = "analyze_image"
    EXPORT_DATA = "export_data"
    MODIFY_SETTINGS = "modify_settings"
    CREATE_USER = "create_user"
    DELETE_USER = "delete_user"
    SYSTEM_ERROR = "system_error"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"

@dataclass
class User:
    """Estrutura de usuário"""
    user_id: str
    username: str
    role: UserRole
    email: str
    full_name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class AuditEvent:
    """Evento de auditoria"""
    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    timestamp: datetime
    ip_address: str
    details: Dict[str, Any]
    success: bool
    risk_level: int  # 0-10

class SecurityManager:
    """
    Gerenciador de segurança central
    Implementa autenticação, autorização, criptografia e auditoria
    Conformidade com HIPAA, GDPR e outras regulamentações
    """
    
    def __init__(self, config_path: str = "security_config.json"):
        self.config = self._load_config(config_path)
        self.db_path = Path(self.config.get('audit_db_path', 'audit.db'))
        self.encryption_key = self._get_or_create_key()
        self.fernet = Fernet(self.encryption_key)
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.sessions = {}  # Active sessions
        self._lock = threading.Lock()
        
        # Inicializar banco de dados
        self._init_database()
        
        # Configurações de segurança
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=8)
        self.password_min_length = 12
        self.password_require_special = True
        self.password_require_numbers = True
        self.password_require_uppercase = True
        
        logger.info("SecurityManager inicializado")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração de segurança"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Configuração padrão
            default_config = {
                'audit_db_path': 'audit.db',
                'encryption_key_path': 'encryption.key',
                'jwt_secret': secrets.token_urlsafe(32),
                'enable_2fa': True,
                'session_timeout_minutes': 480,
                'password_policy': {
                    'min_length': 12,
                    'require_special': True,
                    'require_numbers': True,
                    'require_uppercase': True,
                    'max_age_days': 90
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            return default_config
    
    def _get_or_create_key(self) -> bytes:
        """Obtém ou cria chave de criptografia"""
        key_path = Path(self.config.get('encryption_key_path', 'encryption.key'))
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Gerar nova chave
            key = Fernet.generate_key()
            
            # Salvar com permissões restritas
            with open(key_path, 'wb') as f:
                f.write(key)
            
            # Restringir permissões (apenas leitura para owner)
            os.chmod(key_path, 0o400)
            
            return key
    
    def _init_database(self):
        """Inicializa banco de dados de auditoria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de usuários
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                created_at TEXT,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                totp_secret TEXT
            )
        ''')
        
        # Tabela de eventos de auditoria
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                ip_address TEXT,
                details TEXT,
                success INTEGER,
                risk_level INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Tabela de sessões
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_user ON sessions(user_id)')
        
        conn.commit()
        conn.close()
    
    def create_user(self,
                   username: str,
                   password: str,
                   role: UserRole,
                   email: str,
                   full_name: str,
                   created_by: str) -> Tuple[bool, str]:
        """
        Cria novo usuário com validações de segurança
        
        Args:
            username: Nome de usuário
            password: Senha
            role: Papel do usuário
            email: Email
            full_name: Nome completo
            created_by: ID do usuário que está criando
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        # Validar senha
        is_valid, message = self._validate_password(password)
        if not is_valid:
            return False, message
        
        # Validar username
        if not re.match(r'^[a-zA-Z0-9_]{3,30}$', username):
            return False, "Username inválido. Use apenas letras, números e _"
        
        # Validar email
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False, "Email inválido"
        
        # Hash da senha
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Criar usuário
        user_id = secrets.token_urlsafe(16)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (
                    user_id, username, password_hash, role, email, 
                    full_name, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, password_hash.decode('utf-8'), 
                role.value, email, full_name, 
                datetime.now().isoformat(), 1
            ))
            
            conn.commit()
            conn.close()
            
            # Auditar criação
            self.audit_event(
                AuditEventType.CREATE_USER,
                created_by,
                "0.0.0.0",
                {
                    'new_user_id': user_id,
                    'username': username,
                    'role': role.value
                },
                True
            )
            
            logger.info(f"Usuário criado: {username}")
            return True, user_id
            
        except sqlite3.IntegrityError:
            return False, "Username já existe"
        except Exception as e:
            logger.error(f"Erro ao criar usuário: {str(e)}")
            return False, "Erro ao criar usuário"
    
    def authenticate(self, 
                    username: str, 
                    password: str,
                    ip_address: str = "0.0.0.0") -> Optional[str]:
        """
        Autentica usuário
        
        Args:
            username: Nome de usuário
            password: Senha
            ip_address: Endereço IP
            
        Returns:
            Token JWT se autenticado, None caso contrário
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buscar usuário
        cursor.execute('''
            SELECT user_id, password_hash, is_active, failed_login_attempts,
                   locked_until, role
            FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        
        if not result:
            self.audit_event(
                AuditEventType.LOGIN,
                None,
                ip_address,
                {'username': username, 'reason': 'user_not_found'},
                False,
                risk_level=5
            )
            return None
        
        user_id, password_hash, is_active, failed_attempts, locked_until, role = result
        
        # Verificar se conta está ativa
        if not is_active:
            self.audit_event(
                AuditEventType.LOGIN,
                user_id,
                ip_address,
                {'reason': 'account_disabled'},
                False,
                risk_level=7
            )
            return None
        
        # Verificar bloqueio
        if locked_until:
            locked_until_dt = datetime.fromisoformat(locked_until)
            if datetime.now() < locked_until_dt:
                self.audit_event(
                    AuditEventType.LOGIN,
                    user_id,
                    ip_address,
                    {'reason': 'account_locked'},
                    False,
                    risk_level=8
                )
                return None
            else:
                # Desbloquear
                cursor.execute('''
                    UPDATE users SET locked_until = NULL, failed_login_attempts = 0
                    WHERE user_id = ?
                ''', (user_id,))
        
        # Verificar senha
        if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            # Login bem-sucedido
            cursor.execute('''
                UPDATE users SET last_login = ?, failed_login_attempts = 0
                WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            
            # Criar token JWT
            token = self._create_token(user_id, role)
            
            # Criar sessão
            self._create_session(user_id, token, ip_address)
            
            # Auditar
            self.audit_event(
                AuditEventType.LOGIN,
                user_id,
                ip_address,
                {'success': True},
                True
            )
            
            return token
        else:
            # Senha incorreta
            failed_attempts += 1
            
            # Bloquear se exceder tentativas
            if failed_attempts >= self.max_login_attempts:
                locked_until_dt = datetime.now() + self.lockout_duration
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = ?, locked_until = ?
                    WHERE user_id = ?
                ''', (failed_attempts, locked_until_dt.isoformat(), user_id))
                
                # Alerta de segurança
                self.audit_event(
                    AuditEventType.DATA_BREACH_ATTEMPT,
                    user_id,
                    ip_address,
                    {'reason': 'max_login_attempts_exceeded'},
                    False,
                    risk_level=9
                )
            else:
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = ?
                    WHERE user_id = ?
                ''', (failed_attempts, user_id))
            
            conn.commit()
            conn.close()
            
            self.audit_event(
                AuditEventType.LOGIN,
                user_id,
                ip_address,
                {'reason': 'invalid_password', 'attempt': failed_attempts},
                False,
                risk_level=6
            )
            
            return None
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """Valida senha conforme política"""
        if len(password) < self.password_min_length:
            return False, f"Senha deve ter pelo menos {self.password_min_length} caracteres"
        
        if self.password_require_uppercase and not re.search(r'[A-Z]', password):
            return False, "Senha deve conter pelo menos uma letra maiúscula"
        
        if self.password_require_numbers and not re.search(r'\d', password):
            return False, "Senha deve conter pelo menos um número"
        
        if self.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Senha deve conter pelo menos um caractere especial"
        
        # Verificar senhas comuns
        common_passwords = ['password', '123456', 'admin', 'letmein', 'welcome']
        if password.lower() in common_passwords:
            return False, "Senha muito comum. Escolha uma senha mais segura"
        
        return True, "Senha válida"
    
    def _create_token(self, user_id: str, role: str) -> str:
        """Cria token JWT"""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + self.session_timeout,
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # Token ID único
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verifica e decodifica token JWT"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Verificar se sessão ainda está ativa
            if self._is_session_active(payload['jti']):
                return payload
            else:
                return None
                
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token inválido")
            return None
    
    def _create_session(self, user_id: str, token: str, ip_address: str):
        """Cria sessão de usuário"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload['jti']
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (
                    session_id, user_id, created_at, expires_at, 
                    ip_address, is_active
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id, user_id, datetime.now().isoformat(),
                (datetime.now() + self.session_timeout).isoformat(),
                ip_address, 1
            ))
            
            conn.commit()
            conn.close()
            
            # Adicionar à memória
            with self._lock:
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': datetime.now(),
                    'ip_address': ip_address
                }
                
        except Exception as e:
            logger.error(f"Erro ao criar sessão: {str(e)}")
    
    def _is_session_active(self, session_id: str) -> bool:
        """Verifica se sessão está ativa"""
        # Verificar em memória primeiro
        with self._lock:
            if session_id in self.sessions:
                return True
        
        # Verificar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT is_active, expires_at 
            FROM sessions 
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            is_active, expires_at = result
            expires_dt = datetime.fromisoformat(expires_at)
            
            return is_active and datetime.now() < expires_dt
        
        return False
    
    def authorize(self, token: str, required_role: UserRole) -> bool:
        """
        Verifica autorização baseada em papel
        
        Args:
            token: Token JWT
            required_role: Papel necessário
            
        Returns:
            True se autorizado
        """
        payload = self.verify_token(token)
        
        if not payload:
            return False
        
        user_role = UserRole(payload['role'])
        
        # Hierarquia de papéis
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.TECHNICIAN: 1,
            UserRole.RADIOLOGIST: 2,
            UserRole.PHYSICIAN: 2,
            UserRole.ADMIN: 3,
            UserRole.SYSTEM: 4
        }
        
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Criptografa dados sensíveis
        
        Args:
            data: Dados a criptografar
            
        Returns:
            Dados criptografados
        """
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Descriptografa dados
        
        Args:
            encrypted_data: Dados criptografados
            
        Returns:
            Dados descriptografados
        """
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Erro ao descriptografar: {str(e)}")
            raise
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None):
        """Criptografa arquivo"""
        file_path = Path(file_path)
        
        if not output_path:
            output_path = file_path.with_suffix(file_path.suffix + '.enc')
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Arquivo criptografado: {output_path}")
    
    def decrypt_file(self, file_path: str, output_path: Optional[str] = None):
        """Descriptografa arquivo"""
        file_path = Path(file_path)
        
        if not output_path:
            output_path = file_path.with_suffix('')
        
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        data = self.decrypt_data(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(data)
        
        logger.info(f"Arquivo descriptografado: {output_path}")
    
    def hash_data(self, data: str) -> str:
        """Cria hash seguro de dados"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def audit_event(self,
                   event_type: AuditEventType,
                   user_id: Optional[str],
                   ip_address: str,
                   details: Dict[str, Any],
                   success: bool,
                   risk_level: int = 0):
        """
        Registra evento de auditoria
        
        Args:
            event_type: Tipo do evento
            user_id: ID do usuário
            ip_address: Endereço IP
            details: Detalhes do evento
            success: Se foi bem-sucedido
            risk_level: Nível de risco (0-10)
        """
        event_id = secrets.token_urlsafe(16)
        timestamp = datetime.now()
        
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_events (
                event_id, event_type, user_id, timestamp, 
                ip_address, details, success, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, event_type.value, user_id, timestamp.isoformat(),
            ip_address, json.dumps(details), int(success), risk_level
        ))
        
        conn.commit()
        conn.close()
        
        # Log também no arquivo
        log_msg = f"AUDIT: {event_type.value} | User: {user_id} | IP: {ip_address} | " \
                 f"Success: {success} | Risk: {risk_level} | Details: {details}"
        
        if risk_level >= 7:
            logger.critical(log_msg)
        elif risk_level >= 5:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Alertas em tempo real para eventos críticos
        if risk_level >= 8:
            self._send_security_alert(event_type, user_id, ip_address, details)
    
    def _send_security_alert(self,
                           event_type: AuditEventType,
                           user_id: str,
                           ip_address: str,
                           details: Dict):
        """Envia alerta de segurança"""
        # Implementar notificação (email, SMS, etc)
        logger.critical(f"ALERTA DE SEGURANÇA: {event_type.value} - {details}")
    
    def get_audit_logs(self,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      user_id: Optional[str] = None,
                      event_type: Optional[AuditEventType] = None,
                      min_risk_level: int = 0) -> List[Dict]:
        """
        Recupera logs de auditoria
        
        Args:
            start_date: Data inicial
            end_date: Data final
            user_id: Filtrar por usuário
            event_type: Filtrar por tipo de evento
            min_risk_level: Nível mínimo de risco
            
        Returns:
            Lista de eventos
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        query += " AND risk_level >= ?"
        params.append(min_risk_level)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'event_id': row[0],
                'event_type': row[1],
                'user_id': row[2],
                'timestamp': row[3],
                'ip_address': row[4],
                'details': json.loads(row[5]),
                'success': bool(row[6]),
                'risk_level': row[7]
            })
        
        conn.close()
        
        return events
    
    def check_compliance(self) -> Dict[str, Any]:
        """
        Verifica conformidade com regulamentações (HIPAA, GDPR, etc)
        
        Returns:
            Relatório de conformidade
        """
        compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'compliant': True,
            'issues': [],
            'recommendations': []
        }
        
        # Verificar criptografia
        if not Path(self.config.get('encryption_key_path', '')).exists():
            compliance_report['compliant'] = False
            compliance_report['issues'].append("Chave de criptografia não encontrada")
        
        # Verificar política de senhas
        if self.password_min_length < 8:
            compliance_report['compliant'] = False
            compliance_report['issues'].append("Comprimento mínimo de senha muito baixo")
        
        # Verificar logs de auditoria
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verificar retenção de logs (HIPAA requer 6 anos)
        cursor.execute('''
            SELECT MIN(timestamp) FROM audit_events
        ''')
        
        oldest_log = cursor.fetchone()[0]
        if oldest_log:
            oldest_date = datetime.fromisoformat(oldest_log)
            if (datetime.now() - oldest_date).days > 2190:  # 6 anos
                compliance_report['recommendations'].append(
                    "Considere arquivar logs antigos"
                )
        
        # Verificar tentativas de breach
        cursor.execute('''
            SELECT COUNT(*) FROM audit_events 
            WHERE event_type = ? AND timestamp > ?
        ''', (
            AuditEventType.DATA_BREACH_ATTEMPT.value,
            (datetime.now() - timedelta(days=30)).isoformat()
        ))
        
        breach_attempts = cursor.fetchone()[0]
        if breach_attempts > 10:
            compliance_report['issues'].append(
                f"Alto número de tentativas de breach: {breach_attempts} nos últimos 30 dias"
            )
        
        conn.close()
        
        # Verificar configurações de sessão
        if self.session_timeout > timedelta(hours=12):
            compliance_report['recommendations'].append(
                "Considere reduzir tempo de sessão para melhorar segurança"
            )
        
        return compliance_report
    
    def generate_security_report(self, output_path: str):
        """Gera relatório de segurança detalhado"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system': 'MedAI Security System',
            'compliance_check': self.check_compliance(),
            'user_statistics': self._get_user_statistics(),
            'event_statistics': self._get_event_statistics(),
            'risk_assessment': self._assess_current_risks()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Relatório de segurança gerado: {output_path}")
    
    def _get_user_statistics(self) -> Dict:
        """Obtém estatísticas de usuários"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total de usuários por papel
        cursor.execute('''
            SELECT role, COUNT(*) FROM users 
            WHERE is_active = 1 
            GROUP BY role
        ''')
        
        stats['users_by_role'] = dict(cursor.fetchall())
        
        # Usuários bloqueados
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE locked_until IS NOT NULL 
            AND locked_until > ?
        ''', (datetime.now().isoformat(),))
        
        stats['locked_users'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
    
    def _get_event_statistics(self) -> Dict:
        """Obtém estatísticas de eventos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Eventos por tipo (últimos 30 dias)
        cursor.execute('''
            SELECT event_type, COUNT(*) FROM audit_events 
            WHERE timestamp > ? 
            GROUP BY event_type
        ''', ((datetime.now() - timedelta(days=30)).isoformat(),))
        
        stats['events_by_type'] = dict(cursor.fetchall())
        
        # Taxa de sucesso de login
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success,
                COUNT(*) as total
            FROM audit_events 
            WHERE event_type = ? AND timestamp > ?
        ''', (
            AuditEventType.LOGIN.value,
            (datetime.now() - timedelta(days=7)).isoformat()
        ))
        
        success, total = cursor.fetchone()
        stats['login_success_rate'] = success / total if total > 0 else 0
        
        conn.close()
        
        return stats
    
    def _assess_current_risks(self) -> Dict:
        """Avalia riscos atuais de segurança"""
        risks = {
            'overall_risk_level': 'LOW',
            'factors': []
        }
        
        # Verificar eventos de alto risco recentes
        high_risk_events = self.get_audit_logs(
            start_date=datetime.now() - timedelta(days=1),
            min_risk_level=7
        )
        
        if len(high_risk_events) > 5:
            risks['overall_risk_level'] = 'HIGH'
            risks['factors'].append({
                'factor': 'high_risk_events',
                'count': len(high_risk_events),
                'severity': 'HIGH'
            })
        elif len(high_risk_events) > 0:
            risks['overall_risk_level'] = 'MEDIUM'
            risks['factors'].append({
                'factor': 'high_risk_events',
                'count': len(high_risk_events),
                'severity': 'MEDIUM'
            })
        
        return risks


def require_auth(required_role: UserRole):
    """Decorator para exigir autenticação e autorização"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Obter token do contexto (exemplo)
            token = kwargs.get('auth_token')
            
            if not token:
                raise PermissionError("Token de autenticação não fornecido")
            
            # Verificar autorização
            security_manager = kwargs.get('security_manager')
            if not security_manager:
                raise RuntimeError("SecurityManager não configurado")
            
            if not security_manager.authorize(token, required_role):
                # Auditar tentativa não autorizada
                payload = security_manager.verify_token(token)
                user_id = payload['user_id'] if payload else None
                
                security_manager.audit_event(
                    AuditEventType.ACCESS_DENIED,
                    user_id,
                    "0.0.0.0",
                    {
                        'required_role': required_role.value,
                        'function': func.__name__
                    },
                    False,
                    risk_level=6
                )
                
                raise PermissionError(f"Permissão negada. Papel necessário: {required_role.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
