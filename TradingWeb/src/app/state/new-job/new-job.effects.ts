import { Actions, createEffect, ofType } from "@ngrx/effects";
import { Injectable } from "@angular/core";
import { newJobActions as newJobActions } from "./new-job.actions";
import { catchError, map, switchMap, withLatestFrom } from "rxjs/operators";
import { of } from "rxjs";
import { ConfigValidationService } from "../../services/configValidation.service";
import { ValidationResult } from "../../services/configValidation.service";
import { JobService } from "../../services/jobs.service";
import { select, Store } from "@ngrx/store";
import { getJob, selectNewJobState } from "./new-job.selectors";
import { jobsActions } from "../jobs/jobs.actions";

@Injectable()
export class NewJobEffects{

    validateConfig$ = createEffect(() => this.actions$.pipe(
        ofType(newJobActions.validateConfig),
        switchMap((action) =>
            this.validationService.validateConfig(action.config).pipe(
                map(result => {
                    if (result.config === null){
                        return newJobActions.validateConfigFailure(
                            { error: result.error });
                    }
                    else {
                        return newJobActions.validateConfigSuccess(
                            { config: result.config });
                    }
                }),
                catchError(error => of(newJobActions.validateConfigFailure(
                    { error }))
        )))));

    createJob$ = createEffect(() => this.actions$.pipe(
        ofType(newJobActions.createJob),
        withLatestFrom(this.store.select(getJob)),
        switchMap(([action, job]) => this.jobService.createJob(
            job.name, job.description, job.config).pipe(
                map((job) =>{
                    this.store.dispatch(jobsActions.addJob({ job }));
                    return newJobActions.createJobSuccess();
                }),
                catchError(error => of(newJobActions.createJobFailure(
                  { error }))
        )))));


    constructor(
        private actions$: Actions,
        private validationService: ConfigValidationService,
        private jobService: JobService,
        private store: Store
    ){}
}