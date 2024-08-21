import { ApplicationConfig, InjectionToken } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideHttpClient } from '@angular/common/http';

import { API_URL } from '../constants';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

export const API_URL_TOKEN = new InjectionToken<string>('api_url');

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    { provide: API_URL_TOKEN, useValue: API_URL},
    provideHttpClient(), provideAnimationsAsync()
  ]
};
