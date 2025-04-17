export interface AcademicPaper {
  title: string;
  year: number;
  authors: string[];
  source: string;
  doi: string;
  url: string;
}

export interface CodeExample {
  title: string;
  language: string;
  code: string;
}

export interface Implementation {
  description: string;
  codeExamples: CodeExample[];
}

export interface Category {
  id: string;
  name: string;
  description: string;
}

export interface Heuristic {
  title: string;
  description: string;
  links: {
    title: string;
    url: string;
    type?: 'wikipedia' | 'academic' | 'tutorial';
  }[];
  relationships?: Array<{
    heuristic: string;
    description: string;
    system: string;
  }>;
}

export interface SystemDetails {
  academicFoundations: AcademicPaper[];
  heuristics: Heuristic[];
  implementation: Implementation;
}

export interface DocumentationSystem {
  id: string;
  title: string;
  content: string;
  category: string;
  score: number;
  tags?: string[];
  lastUpdated?: string;
  details?: SystemDetails;
}

export interface Documentation {
  systems: DocumentationSystem[];
  categories: Category[];
}

export async function loadDocumentation(): Promise<Documentation> {
  const response = await fetch('/data/documentation.json');
  if (!response.ok) {
    throw new Error('Failed to load documentation');
  }
  return response.json();
}

export const documentation: Documentation = {
  systems: [
    {
      id: 'api',
      title: 'GoodRobot API',
      content: `The GoodRobot API provides a powerful interface for interacting with the sequence optimization system. Built using FastAPI, it offers comprehensive endpoints for task sequence optimization and pattern analysis.`,
      category: 'integration',
      score: 0.9,
      tags: ['api', 'integration', 'fastapi', 'sequence-optimization'],
      lastUpdated: new Date().toISOString(),
      details: {
        academicFoundations: [],
        heuristics: [
          {
            title: 'API Design Principles',
            description: 'The API follows RESTful principles and provides comprehensive documentation through Swagger UI and ReDoc.',
            links: [
              {
                title: 'FastAPI Documentation',
                url: 'https://fastapi.tiangolo.com/',
                type: 'tutorial'
              }
            ]
          }
        ],
        implementation: {
          description: 'The API is implemented using FastAPI and provides endpoints for sequence optimization and pattern analysis.',
          codeExamples: [
            {
              title: 'Starting the API Server',
              language: 'bash',
              code: 'python run_api.py'
            },
            {
              title: 'Python Client Example',
              language: 'python',
              code: `import requests

response = requests.post(
    "http://localhost:8000/api/sequence/optimize",
    json={
        "tasks": [
            {
                "id": "task1",
                "priority": 0.8,
                "dependencies": [],
                "resources": ["resource1"],
                "estimated_duration": 3600
            }
        ],
        "constraints": {
            "max_duration": 86400,
            "resource_limits": {
                "resource1": 1
            }
        }
    }
)`
            }
          ]
        }
      }
    }
  ],
  categories: [
    {
      id: 'integration',
      name: 'Integration',
      description: 'Documentation for integrating with GoodRobot systems'
    }
  ]
};
