import { createReducer, on } from "@ngrx/store";
import { Job } from "../../models/job.interface";
import { inject } from "@angular/core";
import { Client } from "../../models/client.interface";
import { clientActions } from "./clients.actions";

export interface ClientsState {
    clients: Client[];
    status: 'idle' | 'deleting' | 'loading' | 'error';
    error: string | null;
};

const initialState: ClientsState = {
    clients: [],
    status: 'idle',
    error: null
};


export const clientsReducer = createReducer(
    initialState,
    on(clientActions.load,
        (state) => ({ ...state, status: 'loading' as const })),
    on(clientActions.loadSuccess,
        (state, { clients }) => ({ ...state, clients, status: 'idle' as const })),
    on(clientActions.loadFailure,
        (state, { error }) => ({ ...state, error, status: 'error' as const })),
);