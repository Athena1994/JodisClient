import { createAction, createActionGroup, emptyProps, props } from "@ngrx/store";
import { Client } from "../../models/client.interface";


export const clientActions = createActionGroup({
    source: 'Clients',
    events: {
      Load: emptyProps(),
      LoadSuccess: props<{ clients: Client[] }>(),
      LoadFailure: props<{ error: string }>(),
    }
  });
