# ui/styles.py

CSS_STYLES = """
<style>
    /* --- Base & Layout --- */
    .stApp {
        /* background-color: #ffffff; */ /* White background */
    }
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px; /* Space between tabs */
	}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
		background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
  		background-color: #0d6efd; /* Blue for active tab */
        color: white;
	}
    /* --- Sidebar --- */
    .stSidebar {
        /* background-color: #f8f9fa; /* Light gray background */
        /* border-right: 1px solid #dee2e6; */
    }
    .stSidebar .stButton button {
        background-color: #0d6efd; /* Blue buttons */
        color: white;
        border-radius: 0.3rem;
        transition: background-color 0.3s ease;
    }
    .stSidebar .stButton button:hover {
        background-color: #0b5ed7; /* Darker blue on hover */
    }
    .stSidebar .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 0.3rem;
    }

    /* --- Chat --- */
    .stChatMessage {
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid transparent; /* Base border */
    }
    .stChatMessage[data-testid="chat-message-container-user"] {
        background-color: #e7f3ff; /* Light blue for user messages */
        border-left: 5px solid #0d6efd; /* Blue accent */
    }
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background-color: #f8f9fa; /* Light gray for assistant messages */
        border-left: 5px solid #ffc107; /* Yellow accent */
    }

    /* --- Sources & Details --- */
    .source-box {
        border-left: 4px solid #ffc107; /* Yellow accent for sources */
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        background-color: #fffbeb; /* Very light yellow background */
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .dossier-details-box {
        border: 1px solid #cfe2ff; /* Light blue border */
        background-color: #f2f7ff; /* Very light blue background */
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
     .dossier-active-banner {
        background-color: #d1ecf1; /* Light cyan background */
        color: #0c5460; /* Dark cyan text */
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #bee5eb;
        display: flex;
        align-items: center;
        font-size: 0.95rem;
    }
    .dossier-active-banner span {
        margin-left: 10px;
    }

    /* --- Status Indicators --- */
    .system-status-badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block; /* To allow margin */
        margin-top: 10px; /* Align better with title */
    }
    .system-online {
        background-color: #d4edda; /* Green */
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .system-offline {
        background-color: #f8d7da; /* Red */
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
     .system-initializing {
        background-color: #fff3cd; /* Yellow */
        color: #856404;
        border: 1px solid #ffeeba;
    }

    /* --- Footer --- */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0d6efd; /* Blue footer */
        color: white; /* White text */
        padding: 8px 0;
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #0b5ed7;
        z-index: 99; /* Ensure it's above other elements */
    }
    .footer a {
        color: #ffc107; /* Yellow links */
        text-decoration: none;
    }
     .footer a:hover {
        text-decoration: underline;
    }

    /* --- Misc --- */
     h1, h2, h3 {
        color: #0d6efd; /* Blue headers */
     }
</style>
"""

FOOTER_HTML = """
<div class="footer">
    © 2025 KAP Numérique - Assistant RAG v1.0
    <a href="mailto:support@kap-numerique.fr" target="_blank">Assistance</a>
</div>
"""