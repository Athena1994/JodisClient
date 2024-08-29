import { createFeature, createFeatureSelector, createSelector } from "@ngrx/store";
import { ClientsState } from "./clients.reducer";


export const selectJobsState = createFeatureSelector<ClientsState>('clients');

export const selectClients = createSelector(selectJobsState,
    state => state.clients
);
