import { createAction, createActionGroup, emptyProps, props } from "@ngrx/store";
import { Job } from "../../models/job.interface";


export const newJobActions = createActionGroup({
    source: 'NewJob',
    events: {
      ValidateConfig: props<{ config: string }>(),
      ValidateConfigSuccess: props<{ config: JSON | null }>(),
      ValidateConfigFailure: props<{ error: string }>(),

      SetName: props<{ name: string }>(),
      SetDescription: props<{ description: string }>(),

      CreateJob: emptyProps(),
      CreateJobSuccess: emptyProps(),
      CreateJobFailure: props<{ error: string }>(),
    }
  });
