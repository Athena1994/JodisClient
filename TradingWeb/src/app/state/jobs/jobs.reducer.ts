import { createReducer, on } from "@ngrx/store";
import { Job } from "../../models/job.interface";
import { JobService } from "../../services/jobs.service";
import { inject } from "@angular/core";
import { jobsActions } from "./jobs.actions";

export interface JobsState {
    jobs: Job[];
    selectedJob: Job | null;
    status: 'idle' | 'deleting' | 'loading' | 'loaded' | 'error';
    error: string | null;
};

const initialState: JobsState = {
    jobs: [],
    selectedJob: null,
    status: 'idle',
    error: null
};


export const jobsReducer = createReducer(
    initialState,
    on(jobsActions.load,
        (state) => ({ ...state, status: 'loading' as const })),
    on(jobsActions.loadSuccess,
        (state, { jobs }) => ({ ...state, jobs, status: 'loaded' as const })),
    on(jobsActions.loadFailure,
        (state, { error }) => ({ ...state, error, status: 'error' as const })),
    on(jobsActions.selectJob,
        (state, { job }) => ({ ...state, selectedJob: job })),
    on(jobsActions.addJob,
        (state, { job }) => ({ ...state, jobs: [...state.jobs, job] })),
    on(jobsActions.deleteJobs,
        (state) => ({ ...state, status: 'loading' as const })),
    on(jobsActions.deleteJobsSuccess,
        (state, { deletedIds }) => ({ ...state, jobs: state.jobs.filter(j => !deletedIds.includes(j.id)) })),
    on(jobsActions.deleteJobsFailure,
        (state, { error }) => ({ ...state, error, status: 'error' as const })),
);