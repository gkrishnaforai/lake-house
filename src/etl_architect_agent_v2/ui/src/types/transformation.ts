export interface TransformationTool {
    id: string;
    name: string;
    description: string;
    type: string;
    prompt_template: string;
    default_config: {
        parameters: {
            categories?: string[];
            thresholds?: {
                [key: string]: number;
            };
            aspects?: string[];
        };
    };
    example_input: string;
    example_output: string;
}

export interface TransformationResult {
    status: string;
    message: string;
    new_columns: Array<{
        name: string;
        type: string;
    }>;
    preview_data: Array<{
        [key: string]: any;
    }>;
} 