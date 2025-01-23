new = '''
<style>
/* Modern Typography */
h1#ai4researchers {
    font-family: 'Montserrat', sans-serif;
    font-size: 72px;
    background: linear-gradient(120deg, #6b46c1, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    transition: transform 0.3s ease;
}

h1#ai4researchers:hover {
    transform: scale(1.02);
}

/* Streamlit Elements Styling */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #6b46c1, #9f7aea);
    color: #fff;
    border-radius: 10px;
    border: 2px dashed #fff;
    transition: all 0.3s ease;
}

[data-testid="stFileUploadDropzone"]:hover {
    background: linear-gradient(135deg, #553c9a, #805ad5);
    border-color: #e9d8fd;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(107, 70, 193, 0.2);
}

/* Processing Animation */
@keyframes processing {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.processing {
    animation: processing 2s infinite;
    background: linear-gradient(90deg, #6b46c1, #9f7aea);
    border-radius: 4px;
    padding: 8px 16px;
    color: white;
}

/* Button Styling */
button.css-aqt9oe.edgvbvh10 {
    background: #6b46c1;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    transition: all 0.3s ease;
}

button.css-aqt9oe.edgvbvh10:hover {
    background: #553c9a;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(107, 70, 193, 0.2);
}

/* General Text Styling */
p {
    font-size: 16px;
    color: #2d3748;
    line-height: 1.6;
}

/* Chat Message Styling */
.chat-message {
    display: flex;
    margin: 16px 0;
    padding: 12px;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 8px rgba(107, 70, 193, 0.1);
    transition: transform 0.2s ease;
}

.chat-message:hover {
    transform: translateX(4px);
}

.chat-message .avatar {
    font-size: 24px;
    margin-right: 12px;
    padding: 8px;
    background: #f7fafc;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .message {
    color: #4a5568;
    padding: 8px;
    border-radius: 4px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        🤖
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
    background: linear-gradient(135deg, #fff, #f7fafc);
    background-attachment: fixed;
    min-height: 100vh;
}

/* Glass Effect for Containers */
.stApp > header {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
}

.element-container {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px);
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
'''