# Natural Language Query

A Django web application that enables users to interact with data using natural language queries. The system translates natural language into SQL, retrieves data from a database, and automatically generates appropriate visualizations.

## Features

- Natural language to SQL translation using LLM (GPT-4o-mini)
- Automatic chart generation based on query results
- In-memory DuckDB database with Excel data source

## Setup Instructions

### Prerequisites

- Python 3.10.x
- Required packages (see requirements.txt)
- Github API token for OpenAI API access

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/natural-language-query.git
   cd natural-language-query
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your environment variables:
   - Create a `.env` file in the project root with:
   ```
   GITHUB_TOKEN=your_openai_api_key
   ```

4. Add your data:
   - Place your Excel file in the `nlqapp/data/` directory as `database.xlsx`

5. Run the development server:
   ```
   python manage.py runserver
   ```

6. Access the application at `http://127.0.0.1:8000/`

## Usage

1. Enter a natural language query in the input field (e.g., "What are the average sales by region in 2022")
2. Submit the query
3. View the generated SQL, data results, and visualization

## System Architecture

The application follows a simple architecture:
- **Frontend**: HTML/Bootstrap interface
- **Backend**: Django processing logic
- **Data Layer**: DuckDB in-memory database
- **AI Services**: OpenAI API for natural language processing

### Core Components

- `views.py`: Main Django view handling requests and responses
- `interface.py`: Contains classes for SQL generation and chart creation
- `index.html`: User interface template