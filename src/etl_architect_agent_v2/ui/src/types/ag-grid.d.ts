declare module 'ag-grid-react' {
  import { Component } from 'react';
  export class AgGridReact extends Component<any> {}
}

declare module 'ag-grid-community' {
  export interface IRowNode<TData = any> {
    data: TData;
  }
  
  export interface GridApi {
    forEachNode: (callback: (rowNode: IRowNode, index: number) => void) => void;
    applyTransaction: (transaction: { add?: any[]; remove?: any[]; update?: any[] }) => void;
  }
  
  export interface GridReadyEvent {
    api: GridApi;
  }
  
  export interface ColDef {
    field: string;
    headerName?: string;
    valueFormatter?: (params: ValueFormatterParams) => string;
  }
  
  export interface ValueFormatterParams {
    value: any;
  }
} 