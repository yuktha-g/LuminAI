new = '''
<style>
/* Modern Typography with Purple Theme */
h1#LuminAI {
    font-family: 'Montserrat', sans-serif;
    font-size: 72px;
    background: linear-gradient(120deg, #7c3aed, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titlePulse 3s infinite;
}

@keyframes titlePulse {
    0% { opacity: 0.9; }
    50% { opacity: 1; transform: scale(1.01); }
    100% { opacity: 0.9; }
}

/* Enhanced File Upload Area */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
    border-radius: 12px;
    border: 2px dashed rgba(255, 255, 255, 0.5);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stFileUploadDropzone"]:hover {
    background: linear-gradient(135deg, #6d28d9, #7c3aed);
    border-color: white;
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
}

/* Advanced Loading States */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading {
    background: linear-gradient(90deg, #7c3aed, #a78bfa, #7c3aed);
    background-size: 1000px 100%;
    animation: shimmer 3s infinite;
    border-radius: 8px;
}

/* Interactive Elements */
.interactive-element {
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.interactive-element::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transform: translateX(-100%);
}

.interactive-element:hover::after {
    transform: translateX(100%);
    transition: transform 0.6s;
}

/* Enhanced Chat Interface */
.chat-message {
    display: flex;
    margin: 20px 0;
    padding: 16px;
    border-radius: 12px;
    background: white;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.1);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.15);
}

.chat-message .avatar {
    font-size: 28px;
    margin-right: 16px;
    padding: 12px;
    background: linear-gradient(135deg, #f5f3ff, #ede9fe);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.3s ease;
}

.chat-message:hover .avatar {
    transform: scale(1.1) rotate(5deg);
}

.chat-message .message {
    color: #1f2937;
    padding: 10px;
    border-radius: 8px;
    position: relative;
    z-index: 1;
}

/* Streamlit Specific Overrides */
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
}

.stButton button:hover {
    background: linear-gradient(135deg, #6d28d9, #7c3aed);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
}

/* Glass Morphism Background */
.stApp {
    background: linear-gradient(135deg, #ffffff, #f5f3ff);
}

.stApp > div {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        🌟
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        👤
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

page_bg_img = '''
<style>
body {
    background: linear-gradient(135deg, #ffffff, #f5f3ff);
    background-attachment: fixed;
    min-height: 100vh;
}

/* Enhanced Glass Effect */
.glass-container {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(124, 58, 237, 0.1);
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
}
</style>
'''