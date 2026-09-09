import logging
import sys

# Remove emojis dos logs
class NoEmojiFormatter(logging.Formatter):
    def format(self, record):
        # Remove emojis comuns
        emoji_map = {
            '🧠': '[BRAIN]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📊': '[CHART]',
            '📈': '[GROWTH]',
            '🏥': '[HOSPITAL]',
            '📁': '[FOLDER]',
            '🎯': '[TARGET]',
            '📋': '[CLIPBOARD]',
            '📐': '[METRICS]'
        }
        
        msg = super().format(record)
        for emoji, text in emoji_map.items():
            msg = msg.replace(emoji, text)
        
        return msg

# Configura logging global
def setup_logging(name='MedAI'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Handler para arquivo
    fh = logging.FileHandler(f'{name.lower()}_training.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Handler para console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter sem emojis
    formatter = NoEmojiFormatter(
        '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

