import { Inject, Injectable } from "@angular/core";
import { API_URL_TOKEN } from "../app.config";
import { HttpClient } from "@angular/common/http";
import { catchError, map, Observable, of} from "rxjs";

interface ValidationResponse{
    valid: boolean;
    message: string;
}

export interface ValidationResult{
    config: JSON | null;
    error: string;
}

@Injectable({providedIn: 'root'})
export class ConfigValidationService {

    constructor(private http: HttpClient,
    @Inject(API_URL_TOKEN) private api_url: string) { }

    validateConfig(config: string): Observable<ValidationResult>{

        try {
            const parsedConfig = JSON.parse(config);
            return this.http.post<ValidationResponse>(
                    this.api_url.toString() + "job/validate",
                    parsedConfig
            ).pipe(
                map(response => {
                    if (response.valid){
                        return {
                            config: parsedConfig,
                            error: "",
                        };
                    }
                    else {
                        return {
                            config: null,
                            error: response.message
                        };
                    }
                }),
                catchError((error) => of({
                    config: null,
                    error: error.message,
                }))
            );
        }
        catch(error){
            return of( {
                config: null,
                error: "Input could not be parsed as JSON!"
            });
        }
    }
}