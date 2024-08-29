import { Component } from '@angular/core';
import { Job } from '../../models/job.interface';
import { select, Store } from '@ngrx/store';
import { selectUnassignedJobs } from '../../state/jobs/jobs.selectors';
import { CommonModule } from '@angular/common';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { jobsActions } from '../../state/jobs/jobs.actions';
import { MatSelectModule } from '@angular/material/select';
import { selectClients } from '../../state/clients/clients.selectors';


@Component({
  selector: 'app-unassigned-jobs-control',
  standalone: true,
  imports: [CommonModule, MatCheckboxModule, MatToolbarModule, MatIconModule,
            MatButtonModule, MatSelectModule],
  templateUrl: './unassigned-jobs-control.component.html',
  styleUrl: './unassigned-jobs-control.component.css'
})
export class UnassignedJobsControlComponent {
  jobs: Job[] = [];
  selected: Job[] = [];

  clients$ = this.store.pipe(select(selectClients));

  constructor(private store: Store) {
  }

  assignJobs() {

  }
  deleteJobs() {
    this.store.dispatch(jobsActions.deleteJobs({ jobs: this.selected }));
  }

  updateSelection(job: Job, selected: boolean) {
    if (selected) {
      this.selected.push(job);
    } else {
      this.selected = this.selected.filter(j => j !== job);
    }
  }

  ngOnInit() {
      this.store.pipe(select(selectUnassignedJobs)).subscribe((jobs) => {
        this.jobs = jobs;
        this.selected = this.selected.filter(j => jobs.includes(j));
      });
  }
}
