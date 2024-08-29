export interface Job {
    id: number;
    client_id: number;
    state: string;
    sub_state: string;
    config: JSON;
    name: string;
    description: string;
}