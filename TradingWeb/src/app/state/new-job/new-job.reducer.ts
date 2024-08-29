import { createReducer, on } from "@ngrx/store";
import { Job } from "../../models/job.interface";
import { JobService } from "../../services/jobs.service";
import { inject } from "@angular/core";
import { newJobActions as newJobActions } from "./new-job.actions";

export interface NewJobState {
    name: string | null;
    description: string | null;
    config: JSON | null;
    error: string | null;
    status: 'idle' | 'validating' | 'creating' | 'error';
};

const initialState: NewJobState = {
    name: null,
    description: null,
    config: null,
    status: 'idle',
    error: null
};


export const newJobReducer = createReducer(
    initialState,
    on(newJobActions.setName,
        (state, {name})  => ({ ...state, name })),
    on(newJobActions.setDescription,
        (state, {description}) => ({ ...state, description })),
    on(newJobActions.validateConfig,
        (state) => ({ ...state, config: null,
                                status: 'validating' as const,
                                'error': null })),
    on(newJobActions.validateConfigSuccess,
        (state, { config }) => ({ ...state, config, status: 'idle' as const })),
    on(newJobActions.validateConfigFailure,
        (state, { error }) => ({ ...state, error, status: 'error' as const })),
    on(newJobActions.createJob,
        (state) => ({ ...state, status: 'creating' as const })),
    on(newJobActions.createJobSuccess,
        (state) => initialState),
    on(newJobActions.createJobFailure,
        (state, { error }) => ({ ...state, error, status: 'error' as const }))

);