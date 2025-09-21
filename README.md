# 🤖 FinTracker - AI Financial Assistant 📈

> A revolutionary AI-powered financial chatbot that democratizes trading intelligence for everyone.

Welcome to **FinTracker**! This project combines cutting-edge AI technology with real-time financial data to create an intelligent assistant that makes professional-grade market insights accessible to all investors, from beginners to experts.

-----

## ✨ Core Features

	✅ AI Chatbot: Intelligent financial assistant powered by Google Gemini AI
	✅ Real-time Data: Live stock quotes, market data, and financial news via Alpha Vantage
	✅ Smart Query Processing: Natural language understanding for complex financial queries
	✅ Market Analysis: Top gainers/losers, technical indicators (RSI), and trend analysis
	✅ Multi-format Data: Beautiful tables, charts, and markdown-formatted responses
	✅ Universal Knowledge: Handles both financial data and general educational queries
	✅ Modern UI: Sleek React frontend with dark theme and responsive design
	✅ Team Showcase: Professional About Us page with team member profiles

-----

## 🛠️ Tech Stack

| Component         | Technology                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Backend** | <img src="https://skillicons.dev/icons?i=fastapi" width="20"/> FastAPI, <img src="https://skillicons.dev/icons?i=python" width="20"/> Python 3.11+ |
| **Frontend** | <img src="https://skillicons.dev/icons?i=react" width="20"/> React 18, <img src="https://skillicons.dev/icons?i=js" width="20"/> JavaScript, Styled Components                               |
| **AI Integration** | 🤖 Google Gemini 1.5 Flash, Alpha Vantage MCP                                                                        |
| **APIs** | 📊 Alpha Vantage, Finnhub, Marketaux                                                                   |
| **Styling**| 🎨 Styled Components, Modern Dark Theme                                                                                    |
| **Architecture**| RESTful APIs, CORS-enabled, Microservices Ready                                                                                    |

-----

## 🚀 Getting Started

Ready to experience the future of financial intelligence? Follow these steps!

### Prerequisites

	* Python 3.11+ 
	* Node.js 18+
	* API Keys (see Environment Setup below)

### ⚙️ Environment Setup

1.  **Clone the Repository**
		```sh
		git clone https://github.com/Siddhant-kochhar/hackathon_idea.git
		cd hackathon_idea
		```

2.  **Configure Environment Variables**
		Create a `.env` file in the `app/` directory:

		```env
		# Alpha Vantage MCP Proxy URL
		ALPHA_VANTAGE_MCP_URL=https://mcp.alphavantage.co/mcp?apikey=YOUR_API_KEY

		# Google Gemini AI API Key
		GEMINI_API_KEY=YOUR_GEMINI_API_KEY

		# Optional: Additional API Keys
		FINNHUB_API_KEY=YOUR_FINNHUB_KEY
		MARKETAUX_API_TOKEN=YOUR_MARKETAUX_TOKEN

		# Server Configuration
		PORT=8002
		HOST=0.0.0.0
		DEBUG=True
		```

3.  **Backend Setup** 🐍
		```sh
		cd app
		pip install -r requirements.txt
		python main.py
		```
		Backend runs on: `http://localhost:8002`

4.  **Frontend Setup** ⚛️
		```sh
		cd hackathon_website
		npm install
		npm start
		```
		Frontend runs on: `http://localhost:3000`

-----

## 🧪 Testing the Application

### API Endpoints Testing

```sh
# Test stock quote
curl -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "quote AAPL"}'

# Test market analysis
curl -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "top gainers"}'

# Test AI education
curl -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "explain compound interest"}'
```

### Frontend Testing

	💬 Chatbot Interface: `http://localhost:3000/main`
	👥 About Us Page: `http://localhost:3000/aboutus`
	🏠 Landing Page: `http://localhost:3000/`

-----

## 📚 API Documentation

### Query Examples

	*Stock Data* 📈
	- `"quote AAPL"` - Get current Apple stock price
	- `"daily MSFT"` - Microsoft daily data
	- `"intraday TSLA 5min"` - Tesla 5-minute intervals

	*Market Analysis* 📊
	- `"top gainers"` - Best performing stocks
	- `"top losers"` - Worst performing stocks  
	- `"most active stocks"` - High volume trading

	*Technical Analysis* 🔍
	- `"RSI AAPL daily"` - Relative Strength Index
	- `"AAPL price history"` - Historical price data

	*AI Assistant* 🤖
	- `"explain diversification"` - Financial education
	- `"latest market news"` - AI-powered market insights
	- `"what is compound interest?"` - Educational content

### Built-in UI Tester
Visit `http://localhost:8002/ui` for an interactive API testing interface!

-----

## 📁 Project Structure

```
hackathon_idea/
├── 🤖 app/                    # FastAPI Backend
│   ├── main.py               # Main server with AI integration
│   ├── .env                  # Environment configuration
│   └── requirements.txt      # Python dependencies
├── 🌐 hackathon_website/      # React Frontend
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # React pages (Landing, Main, About)
│   │   └── App.js           # Main React application
│   ├── public/              # Static assets & team images
│   └── package.json         # Node.js dependencies
└── README.md                # You are here! 📍
```

-----

## 🎯 Key Features Showcase

### 🧠 Intelligent Query Processing
Our AI can understand natural language queries like:
- *"Show me Apple's performance last week"*
- *"What are the top tech stocks today?"*
- *"Explain what P/E ratio means"*

### 📊 Beautiful Data Visualization
- **Markdown Tables**: Clean, formatted financial data
- **Real-time Updates**: Live market information
- **Multi-format Support**: CSV, JSON, and human-readable formats

### 🎨 Modern User Experience
- **Dark Theme Design**: Easy on the eyes for long trading sessions
- **Responsive Layout**: Works perfectly on desktop and mobile
- **Smooth Navigation**: React Router for seamless page transitions

-----

## 🌟 Team

Meet the brilliant minds behind FinTracker:

- **Siddhant Kochhar** - Technical Lead & Full-Stack Developer
- **Arpit Gupta** - Frontend Developer & UI/UX Designer  
- **Kush Saini** - Backend Architect & Data Security Specialist

-----

## 🚀 Business Model

**Freemium Strategy:**
- 🆓 **Free Tier**: Real-time quotes, basic analysis, AI chat
- ⭐ **Premium ($9.99/month)**: Advanced alerts, portfolio tracking, premium AI insights

**Target Market:** 10M+ retail traders in India seeking affordable financial intelligence.

-----

## 🎉 Demo & Presentation

This project was built for a hackathon presentation focusing on:
- **Social Impact**: Democratizing financial knowledge
- **Technical Innovation**: AI-powered market intelligence
- **Business Viability**: Sustainable freemium model

-----

## 🙌 Contributing

We welcome contributions! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first to discuss your ideas.

-----

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

**💡 Built with ❤️ by Team FinTracker | Making financial intelligence accessible to everyone!**

