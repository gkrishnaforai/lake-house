import React, { useState, useEffect } from 'react';

type Pipeline = {
  id: string;
  name: string;
  description: string;
  source: string;
  destination: string;
  schedule: string;
  status: 'active' | 'inactive' | 'failed';
  lastRun?: string;
  terraformConfig?: string;
};

type PipelineFormData = {
  name: string;
  description: string;
  source: string;
  destination: string;
  schedule: string;
};

const Infrastructure: React.FC = () => {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState<PipelineFormData>({
    name: '',
    description: '',
    source: '',
    destination: '',
    schedule: 'daily'
  });
  const [selectedPipeline, setSelectedPipeline] = useState<Pipeline | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    // Simulate fetching pipelines
    const fetchPipelines = async () => {
      try {
        // TODO: Replace with actual API call
        const mockPipelines: Pipeline[] = [
          {
            id: '1',
            name: 'Sales Data Pipeline',
            description: 'ETL pipeline for sales data',
            source: 's3://raw-data/sales',
            destination: 's3://processed-data/sales',
            schedule: 'daily',
            status: 'active',
            lastRun: new Date().toISOString()
          }
        ];
        setPipelines(mockPipelines);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPipelines();
  }, []);

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // TODO: Replace with actual API call to create pipeline
      const newPipeline: Pipeline = {
        id: Date.now().toString(),
        ...formData,
        status: 'inactive'
      };
      setPipelines([...pipelines, newPipeline]);
      setShowForm(false);
      setFormData({
        name: '',
        description: '',
        source: '',
        destination: '',
        schedule: 'daily'
      });
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleGenerateTerraform = async (pipeline: Pipeline) => {
    try {
      // TODO: Replace with actual API call to InfrastructureAgent
      const mockTerraform = `
resource "aws_glue_job" "${pipeline.name.toLowerCase().replace(/\s+/g, '_')}" {
  name     = "${pipeline.name}"
  role_arn = "arn:aws:iam::123456789012:role/GlueServiceRole"
  
  command {
    script_location = "s3://glue-scripts/${pipeline.name}.py"
  }
  
  default_arguments = {
    "--job-language" = "python"
    "--job-bookmark-option" = "job-bookmark-enable"
  }
  
  execution_property {
    max_concurrent_runs = 1
  }
}`;
      
      setPipelines(pipelines.map(p => 
        p.id === pipeline.id ? { ...p, terraformConfig: mockTerraform } : p
      ));
    } catch (e: any) {
      setError(e.message);
    }
  };

  if (loading) return <div style={{ padding: 24 }}>Loading infrastructure...</div>;
  if (error) return <div style={{ padding: 24, color: 'red' }}>{error}</div>;

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h1 style={{ margin: 0 }}>ETL Pipelines</h1>
        <button
          onClick={() => setShowForm(true)}
          style={{
            padding: '12px 24px',
            background: '#1a73e8',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer'
          }}
        >
          Create New Pipeline
        </button>
      </div>

      {/* Pipeline List */}
      <div style={{ display: 'grid', gap: 16 }}>
        {pipelines.map(pipeline => (
          <div
            key={pipeline.id}
            style={{
              background: 'white',
              padding: 24,
              borderRadius: 8,
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <h3 style={{ margin: 0 }}>{pipeline.name}</h3>
                <p style={{ color: '#666', margin: '8px 0' }}>{pipeline.description}</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px' }}>
                  <div style={{ color: '#666' }}>Source:</div>
                  <div>{pipeline.source}</div>
                  <div style={{ color: '#666' }}>Destination:</div>
                  <div>{pipeline.destination}</div>
                  <div style={{ color: '#666' }}>Schedule:</div>
                  <div>{pipeline.schedule}</div>
                  <div style={{ color: '#666' }}>Status:</div>
                  <div>
                    <span style={{
                      padding: '4px 8px',
                      borderRadius: 4,
                      background: pipeline.status === 'active' ? '#e6f4ea' : 
                                pipeline.status === 'failed' ? '#fce8e6' : '#fef7e0',
                      color: pipeline.status === 'active' ? '#137333' :
                            pipeline.status === 'failed' ? '#c5221f' : '#b06000'
                    }}>
                      {pipeline.status}
                    </span>
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => handleGenerateTerraform(pipeline)}
                  style={{
                    padding: '8px 16px',
                    background: '#34a853',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  Generate Terraform
                </button>
                <button
                  onClick={() => setSelectedPipeline(pipeline)}
                  style={{
                    padding: '8px 16px',
                    background: '#1a73e8',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  View Logs
                </button>
              </div>
            </div>
            {pipeline.terraformConfig && (
              <div style={{ marginTop: 16, padding: 16, background: '#f8f9fa', borderRadius: 4 }}>
                <h4 style={{ margin: '0 0 8px 0' }}>Terraform Configuration</h4>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{pipeline.terraformConfig}</pre>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Create Pipeline Form */}
      {showForm && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{
            background: 'white',
            padding: 24,
            borderRadius: 8,
            width: '100%',
            maxWidth: 600
          }}>
            <h2 style={{ margin: '0 0 24px 0' }}>Create New Pipeline</h2>
            <form onSubmit={handleFormSubmit}>
              <div style={{ display: 'grid', gap: 16 }}>
                <div>
                  <label style={{ display: 'block', marginBottom: 8 }}>Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={e => setFormData({ ...formData, name: e.target.value })}
                    style={{
                      width: '100%',
                      padding: 8,
                      borderRadius: 4,
                      border: '1px solid #ddd'
                    }}
                    required
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: 8 }}>Description</label>
                  <textarea
                    value={formData.description}
                    onChange={e => setFormData({ ...formData, description: e.target.value })}
                    style={{
                      width: '100%',
                      padding: 8,
                      borderRadius: 4,
                      border: '1px solid #ddd',
                      minHeight: 100
                    }}
                    required
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: 8 }}>Source</label>
                  <input
                    type="text"
                    value={formData.source}
                    onChange={e => setFormData({ ...formData, source: e.target.value })}
                    style={{
                      width: '100%',
                      padding: 8,
                      borderRadius: 4,
                      border: '1px solid #ddd'
                    }}
                    required
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: 8 }}>Destination</label>
                  <input
                    type="text"
                    value={formData.destination}
                    onChange={e => setFormData({ ...formData, destination: e.target.value })}
                    style={{
                      width: '100%',
                      padding: 8,
                      borderRadius: 4,
                      border: '1px solid #ddd'
                    }}
                    required
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: 8 }}>Schedule</label>
                  <select
                    value={formData.schedule}
                    onChange={e => setFormData({ ...formData, schedule: e.target.value })}
                    style={{
                      width: '100%',
                      padding: 8,
                      borderRadius: 4,
                      border: '1px solid #ddd'
                    }}
                  >
                    <option value="hourly">Hourly</option>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                  </select>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 24 }}>
                <button
                  type="submit"
                  style={{
                    padding: '12px 24px',
                    background: '#1a73e8',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  Create Pipeline
                </button>
                <button
                  type="button"
                  onClick={() => setShowForm(false)}
                  style={{
                    padding: '12px 24px',
                    background: '#f8f9fa',
                    color: '#666',
                    border: '1px solid #ddd',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Logs Modal */}
      {selectedPipeline && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{
            background: 'white',
            padding: 24,
            borderRadius: 8,
            width: '100%',
            maxWidth: 800,
            maxHeight: '80vh',
            overflow: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h2 style={{ margin: 0 }}>Pipeline Logs: {selectedPipeline.name}</h2>
              <button
                onClick={() => setSelectedPipeline(null)}
                style={{
                  padding: '8px 16px',
                  background: '#f8f9fa',
                  color: '#666',
                  border: '1px solid #ddd',
                  borderRadius: 4,
                  cursor: 'pointer'
                }}
              >
                Close
              </button>
            </div>
            <div style={{ background: '#f8f9fa', padding: 16, borderRadius: 4 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {logs.length > 0 ? logs.join('\n') : 'No logs available'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Infrastructure; 