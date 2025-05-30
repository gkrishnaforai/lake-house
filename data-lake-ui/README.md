# Data Lake UI

A modern, user-friendly interface for managing and exploring data in an AWS-based data lakehouse. This application provides tools for data cataloging, exploration, and ETL pipeline management.

## Features

### Dashboard

- Overview of system statistics
- Recent activity monitoring
- Quick access to common actions
- System health status

### Data Catalog

- Browse and search tables
- View table schemas and metadata
- Upload and manage data files
- Track data lineage

### Data Explorer

- SQL query interface for Athena
- Query history and saved queries
- Results visualization
- Export capabilities

### Infrastructure

- ETL pipeline management
- Terraform configuration generation
- Pipeline monitoring and logs
- Resource status tracking

## Tech Stack

- **Frontend**: React 18 with TypeScript
- **Routing**: React Router v6
- **Styling**: CSS-in-JS with inline styles
- **Backend Integration**: RESTful API calls to AWS services

## AWS Services Integration

The application integrates with the following AWS services:

- AWS Glue for data catalog and ETL
- Amazon S3 for data storage
- Amazon Athena for SQL querying
- AWS Lambda for serverless functions
- AWS IAM for authentication and authorization

## Getting Started

### Prerequisites

- Node.js 16.x or later
- npm 7.x or later
- AWS account with appropriate permissions

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd data-lake-ui
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory with your AWS configuration:

   ```
   REACT_APP_AWS_REGION=your-region
   REACT_APP_API_ENDPOINT=your-api-endpoint
   ```

4. Start the development server:
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`.

## Development

### Project Structure

```
data-lake-ui/
├── src/
│   ├── components/         # React components
│   ├── api/               # API integration
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   ├── App.tsx           # Main application component
│   └── index.tsx         # Application entry point
├── public/               # Static assets
├── package.json         # Dependencies and scripts
└── tsconfig.json       # TypeScript configuration
```

### Available Scripts

- `npm start`: Runs the app in development mode
- `npm test`: Launches the test runner
- `npm run build`: Builds the app for production
- `npm run eject`: Ejects from Create React App

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
