import { Actions, createEffect, ofType } from "@ngrx/effects";
import { Injectable } from "@angular/core";
import { JobService } from "../../services/jobs.service";
import { catchError, map, switchMap } from "rxjs/operators";
import { of } from "rxjs";
import { ClientService } from "../../services/clients.service";
import { clientActions } from "./clients.actions";


@Injectable()
export class ClientsEffects{

    loadClients$ = createEffect(() => this.actions$.pipe(
        ofType(clientActions.load),
        switchMap(() => this.clientService.getClients().pipe(
            map(clients => clientActions.loadSuccess({ clients })),
            catchError(error => of(clientActions.loadFailure({ error })))
        ))
    ));

    constructor(
        private actions$: Actions,
        private clientService: ClientService
    ){}
}