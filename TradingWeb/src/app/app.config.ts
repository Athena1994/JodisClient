import { ApplicationConfig, InjectionToken } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideHttpClient } from '@angular/common/http';

import { API_URL } from '../constants';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { provideStore } from '@ngrx/store';
import { provideEffects } from '@ngrx/effects';
import { jobsReducer } from './state/jobs/jobs.reducer';
import { JobsEffects } from './state/jobs/jobs.effects';
import { NewJobEffects } from './state/new-job/new-job.effects';
import { newJobReducer } from './state/new-job/new-job.reducer';

import { provideStoreDevtools } from '@ngrx/store-devtools';
import { ClientsEffects } from './state/clients/clients.effects';
import { clientsReducer } from './state/clients/clients.reducer';

export const API_URL_TOKEN = new InjectionToken<string>('api_url');

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    { provide: API_URL_TOKEN, useValue: API_URL },
    provideHttpClient(), provideAnimationsAsync(),
    provideStore({
      jobs: jobsReducer,
      newJob: newJobReducer,
      clients: clientsReducer}),
    provideEffects([JobsEffects, NewJobEffects, ClientsEffects]),
    provideStoreDevtools(
      {maxAge: 25,
      logOnly: false,}
    )
]
};
