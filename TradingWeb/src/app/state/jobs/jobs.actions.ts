import { createAction, createActionGroup, emptyProps, props } from "@ngrx/store";
import { Job } from "../../models/job.interface";


export const jobsActions = createActionGroup({
    source: 'Jobs',
    events: {
      Load: emptyProps(),
      LoadSuccess: props<{ jobs: Job[] }>(),
      LoadFailure: props<{ error: string }>(),

      DeleteJobs: props<{ jobs: Job[] }>(),
      DeleteJobsSuccess: props<{ deletedIds: number[] }>(),
      DeleteJobsFailure: props<{ error: string }>(),

      SelectJob: props<{ job: Job }>(),

      AddJob: props<{ job: Job }>(),
    }
  });
