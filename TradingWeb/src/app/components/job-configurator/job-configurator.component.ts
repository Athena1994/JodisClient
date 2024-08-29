import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { MatInputModule } from '@angular/material/input';
import { TextFieldModule } from '@angular/cdk/text-field';
import { MatFormFieldModule } from '@angular/material/form-field';
import { select, Store } from '@ngrx/store';
import { Job } from '../../models/job.interface';
import { selectSelectedJob } from '../../state/jobs/jobs.selectors';
import { JsonEditorOptions, NgJsonEditorModule } from 'ang-jsoneditor';

@Component({
  selector: 'app-job-configurator',
  standalone: true,
  imports: [CommonModule, TextFieldModule, MatFormFieldModule, MatInputModule,
  NgJsonEditorModule],
  templateUrl: './job-configurator.component.html',
  styleUrl: './job-configurator.component.css'
})
export class JobConfiguratorComponent {

    selectedJob$ = this.store.pipe(select(selectSelectedJob));

    editorOptions: JsonEditorOptions;


    constructor(private store: Store) {
        this.editorOptions = new JsonEditorOptions()
        this.editorOptions.modes = ['code', 'text', 'tree', 'view']; // set all allowed modes

    }

}
