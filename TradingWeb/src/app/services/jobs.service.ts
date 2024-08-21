import { Inject, Injectable} from "@angular/core";
import { Job } from "../models/job.interface";
import { HttpClient } from '@angular/common/http';
import { Observable } from "rxjs";
import { API_URL_TOKEN } from "../app.config";

@Injectable({providedIn: 'root'})
export class JobService {

    constructor(private http: HttpClient,
                @Inject(API_URL_TOKEN) private api_url: string) {}

    getJobs(assigned: boolean,
            unassigned: boolean,
            finished: boolean): Observable<Job[]> {
        const params: any = {};
        if (assigned) {
            params.assigned = true;
        }
        if (unassigned) {
            params.unassigned = true;
        }
        if (finished) {
            params.finished = true;
        }

        return this.http.get<Job[]>(
            this.api_url.toString()+"jobs", {params});
    }
}