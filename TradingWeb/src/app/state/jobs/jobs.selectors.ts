import { createFeature, createFeatureSelector, createSelector } from "@ngrx/store";
import { JobsState } from "./jobs.reducer";


export const selectJobsState = createFeatureSelector<JobsState>('jobs');

export const selectJobs = createSelector(selectJobsState,
    state => state.jobs
);
export const selectAssignedJobs = createSelector(selectJobs,
    jobs => jobs.filter(job => job.state === 'ASSIGNED')
);
export const selectUnassignedJobs = createSelector(selectJobs,
    jobs => jobs.filter(job => job.state === 'UNASSIGNED')
);
export const selectFinishedJobs = createSelector(selectJobs,
    jobs => jobs.filter(job => job.state === 'FINISHED')
);

export const selectSelectedJob = createSelector(selectJobsState,
    state => state.selectedJob
);