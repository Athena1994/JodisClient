import { Actions, createEffect, ofType } from "@ngrx/effects";
import { Injectable } from "@angular/core";
import { JobService } from "../../services/jobs.service";
import { jobsActions } from "./jobs.actions";
import { catchError, map, switchMap } from "rxjs/operators";
import { of } from "rxjs";


@Injectable()
export class JobsEffects{

    loadJobs$ = createEffect(() => this.actions$.pipe(
        ofType(jobsActions.load),
        switchMap(() => this.jobService.getJobs(true, true, true).pipe(
            map(jobs => jobsActions.loadSuccess({ jobs })),
            catchError(error => of(jobsActions.loadFailure({ error })))
        ))
    ));
    deleteJobs = createEffect(() => this.actions$.pipe(
        ofType(jobsActions.deleteJobs),
        switchMap(({ jobs }) => this.jobService.deleteJobs(jobs).pipe(
            map((ids: number[]) => jobsActions.deleteJobsSuccess({ deletedIds: ids })),
            catchError(error => of(jobsActions.deleteJobsFailure({ error })))
        ))
    ));

    constructor(
        private actions$: Actions,
        private jobService: JobService
    ){}
}